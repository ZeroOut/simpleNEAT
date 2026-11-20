#pragma once

#ifndef MYNEAT_CUDA_NN_HPP
#define MYNEAT_CUDA_NN_HPP

// [CUDA] 基于 cuBLAS 的全连接网络批量引擎（原文件是 FCNNFeedForwardPredict 的 CPU 草稿，无 CUDA 代码，已重写）
//
// 设计说明：
// - 设备端激活函数硬编码为 Sigmoid/DerivativeSigmoid（Opts.ActiveFunction 是 std::function，无法在设备端执行）
// - 前向/反向用 strided-batched GEMM：种群 P 个同拓扑 FCNN 的权重打包成 [P × WPG]，一次批量算完
// - 训练（单 genome，P=1）为全批量 mini-batch：所有样本梯度求和取均值后一次更新（与 CPU 在线 SGD
//   数学上有差异；速度对比以"每 epoch 耗时"为准）
// - 交配进化路径：交叉/变异/创新号簿记在 CPU（NEAT 的全局串行结构不适合 GPU），GPU 只承担
//   计算量最大的适应度评估（种群 × 全部样本）
// - 权重打包布局与 lib 完全一致：[W1(h0×I) | W2 | ... | WL(O×hK) | b1 | ... | bL]，行优先、in 最快

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "NeuralNetwork.hpp"

namespace znn {

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
        std::cerr << "[CUDA] error: " << cudaGetErrorString(e_) << " at " << __FILE__ << ":" << __LINE__ << "\n"; exit(1); } } while (0)

#define CUBLAS_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "[cuBLAS] error: " << s_ << " at " << __FILE__ << ":" << __LINE__ << "\n"; exit(1); } } while (0)

    __device__ __forceinline__ float CudaActivate(float x) {
        return 1.f / (1.f + expf(-x));  // 设备端固定 Sigmoid
    }

    // 加偏置+激活：A[idx] = act(A[idx] + bias)。population 模式下按 genome 分组取各自 bias
    __global__ void AddBiasActivateKernel(float *A, const float *biasBase, size_t total, int cols,
                                          size_t genomeElems, size_t biasGenomeStride, size_t biasLayerOffset) {
        size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        size_t c = idx % cols;
        size_t g = idx / genomeElems;
        A[idx] = CudaActivate(A[idx] + biasBase[g * biasGenomeStride + biasLayerOffset + c]);
    }

    // [CUDA] delta 回传核：D_prev[s][i] = Σ_j D_cur[s][j] * W[j][i]（W 为 rm [out×in]）
    // 替代同缓冲 B/C 别名的 GEMM 写法
    __global__ void DeltaPropKernel(const float *dCur, const float *W, float *dPrev, int size, int in, int out) {
        size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        size_t total = size_t(size) * in;
        if (idx >= total) {
            return;
        }
        int sIdx = int(idx / in);
        int i = int(idx % in);
        float sum = 0.f;
        for (int j = 0; j < out; ++j) {
            sum += dCur[sIdx * out + j] * W[size_t(j) * in + i];
        }
        dPrev[sIdx * in + i] = sum;
    }

    // 输出层 delta = (target - out) * f'(out)，f'(o)=o*(1-o)（与 lib 的 delta rule 方向一致）。
    // [FIX] sigmoid 在 float 下会精确饱和到 0/1 → O*(1-O)=0 → 梯度恒零、训练冻结
    // （classification CUDA 版 fitness 恒 0.5 的根因）。把 o 钳进 [1e-6, 1-1e-6] 保证梯度存在
    __global__ void OutputDeltaKernel(const float *out, const float *target, float *delta, size_t total) {
        size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        float o = fminf(1.f - 1e-6f, fmaxf(1e-6f, out[idx]));
        delta[idx] = (target[idx] - o) * o * (1.f - o);
    }

    // 隐藏层 delta *= f'(act)
    __global__ void MulDerivKernel(float *D, const float *A, size_t total) {
        size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        float a = A[idx];
        D[idx] *= a * (1.f - a);
    }

    // gradB[c] = Σ_s D[s][c]：每列一个 block，block 内 256 线程分块求和 + 归约
    __global__ void SumColsKernel(const float *D, float *gradB, int S, int cols) {
        __shared__ float sh[256];
        int c = blockIdx.x;
        if (c >= cols) {
            return;
        }
        float sum = 0.f;
        for (int s = threadIdx.x; s < S; s += blockDim.x) {
            sum += D[(size_t) s * cols + c];
        }
        sh[threadIdx.x] = sum;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) {
                sh[threadIdx.x] += sh[threadIdx.x + off];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            gradB[c] = sh[0];
        }
    }

    // 更新：v = beta*v + lrScale*G；W = clamp(W + v, ±range)（beta=0 退化为纯 SGD，
    // 语义与 lib 的 BPMomentum + 越界截断一致；v 驻留显存，跨轮累积）
    __global__ void ApplyUpdateKernel(float *W, const float *G, float *V, size_t total, float lrScale, float beta, float range) {
        size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        float v = beta * V[idx] + lrScale * G[idx];
        V[idx] = v;
        W[idx] = fminf(range, fmaxf(-range, W[idx] + v));
    }

    // [V3] 按行索引收集：dst 第 i 行 = src[ids[i]] 第 i 行（逐个体模式用——每个个体引用
    // 种群中不同 genome 的权重块，strided-batched GEMM 只支持等距 stride，先在卡上重排）
    __global__ void GatherRowsKernel(const float *src, const int *ids, float *dst, size_t rowFloats, size_t total) {
        size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        if (i >= total) {
            return;
        }
        dst[i] = src[(size_t) ids[i / rowFloats] * rowFloats + i % rowFloats];
    }

    // FCNN 层尺寸表：{I, h0, h1, ..., O}
    inline std::vector<ulong> FCNNLayers() {
        std::vector<ulong> layers;
        layers.push_back(Opts.InputSize);
        for (auto &h: Opts.FCNN_hideLayers) {
            layers.push_back(h);
        }
        layers.push_back(Opts.OutputSize);
        return layers;
    }

    inline size_t FCNNPackedSize(const std::vector<ulong> &layers) {
        size_t n = 0;
        for (size_t l = 1; l < layers.size(); ++l) {
            n += layers[l] * layers[l - 1];
        }
        for (size_t l = 1; l < layers.size(); ++l) {
            n += layers[l];
        }
        return n;
    }

    // 把一个 FCNN genome（lib 布局：connections=[W1|W2|..|WL]，bias=[隐藏层平铺..., 输出层]）打包为
    // [W1 | W2 | ... | WL | b1 | ... | bL]（行优先，in 最快），与 NewFCNN/FCNNBackPropagation 的索引约定一致
    inline std::vector<float> PackFCNNGenome(NetworkGenome &nn) {
        auto layers = FCNNLayers();
        size_t L = layers.size() - 1;
        std::vector<float> packed(FCNNPackedSize(layers));
        size_t off = 0;
        size_t connOff = 0;  // 各层连接块在 genome Connections 中的起始下标

        for (size_t l = 1; l <= L; ++l) {
            ulong in = layers[l - 1], out = layers[l];
            for (ulong j = 0; j < out; ++j) {
                for (ulong i = 0; i < in; ++i) {
                    packed[off++] = nn.Connections[connOff + j * in + i].Weight;
                }
            }
            connOff += out * in;
        }
        // 隐藏层偏置：Neurons[I+O ...] 按层平铺顺序
        ulong hid = Opts.InputSize + Opts.OutputSize;
        for (size_t l = 1; l < L; ++l) {
            for (ulong j = 0; j < layers[l]; ++j) {
                packed[off++] = nn.Neurons[hid++].Bias;
            }
        }
        // 输出层偏置：Neurons[I .. I+O)
        for (ulong ol = 0; ol < layers[L]; ++ol) {
            packed[off++] = nn.Neurons[Opts.InputSize + ol].Bias;
        }
        return packed;
    }

    struct CudaFcnnEngine {
        std::vector<ulong> layers;  // {I, h0, ..., O}
        uint L = 0;                 // 权重矩阵层数 = layers.size()-1
        uint P = 1;                 // 种群大小（训练时为1）
        uint S = 0;                 // 样本数
        size_t WPG = 0;             // 每 genome 打包 float 数
        std::vector<size_t> wOff, bOff, wNum;  // 各层权重/偏置在打包块内的偏移与元素数

        float *dX = nullptr;        // [S×I]
        float *dT = nullptr;        // [S×O]
        float *dPackW = nullptr;    // [P × WPG]
        float *dD = nullptr;        // [2 × S × maxLayer] delta 双缓冲（前半=当前层，后半=上一层写入区）
        size_t maxLayer = 0;        // [CUDA] 最大层宽（TrainEpoch/TrainBatch 的 delta 缓冲寻址用）
        float *dGrad = nullptr;     // 布局同打包块（训练用）
        float *dMom = nullptr;      // [CUDA] 动量速度缓冲（布局同打包块）
        float *dActBuf = nullptr;   // Σ_l P*S*layers[l]（l=1..L）
        std::vector<float *> dAct;  // dActBuf + 各层偏移
        cublasHandle_t handle = nullptr;

        CudaFcnnEngine(const std::vector<ulong> &layers_, uint P_, uint S_, int device = 0) : layers(layers_), P(P_), S(S_) {
            CUDA_CHECK(cudaSetDevice(device));
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            std::cout << "[CUDA] device: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";

            L = uint(layers.size() - 1);
            WPG = FCNNPackedSize(layers);
            wOff.resize(L + 1);
            bOff.resize(L + 1);
            wNum.resize(L + 1);
            size_t wTotal = 0;
            for (size_t l = 1; l <= L; ++l) {
                wOff[l] = wTotal;
                wNum[l] = layers[l] * layers[l - 1];
                wTotal += wNum[l];
            }
            for (size_t l = 1; l <= L; ++l) {
                bOff[l] = wTotal;
                wTotal += layers[l];
            }

            CUDA_CHECK(cudaMalloc(&dX, size_t(S) * layers[0] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dT, size_t(S) * layers[L] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dPackW, P * WPG * sizeof(float)));
            maxLayer = 0;
            for (auto l: layers) {
                maxLayer = std::max(maxLayer, (size_t) l);
            }
            CUDA_CHECK(cudaMalloc(&dD, 2 * size_t(S) * maxLayer * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dGrad, P * WPG * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dMom, P * WPG * sizeof(float)));
            CUDA_CHECK(cudaMemset(dMom, 0, P * WPG * sizeof(float)));

            dActBuf = nullptr;
            size_t actTotal = 0;
            std::vector<size_t> actOffv(L + 1, 0);
            for (size_t l = 1; l <= L; ++l) {
                actOffv[l] = actTotal;
                actTotal += size_t(P) * S * layers[l];
            }
            CUDA_CHECK(cudaMalloc(&dActBuf, actTotal * sizeof(float)));
            dAct.assign(L + 1, nullptr);
            for (size_t l = 1; l <= L; ++l) {
                dAct[l] = dActBuf + actOffv[l];
            }

            CUBLAS_CHECK(cublasCreate(&handle));
        }

        ~CudaFcnnEngine() {
            if (handle) {
                cublasDestroy(handle);
            }
            cudaFree(dX);
            cudaFree(dT);
            cudaFree(dPackW);
            cudaFree(dD);
            cudaFree(dGrad);
            cudaFree(dMom);
            cudaFree(dActBuf);
            cudaFree(dNnId);
            cudaFree(dInputsN);
            cudaFree(dGatherW);
        }

        void UploadX(const std::vector<float> &X) {
            CUDA_CHECK(cudaMemcpy(dX, X.data(), size_t(S) * layers[0] * sizeof(float), cudaMemcpyHostToDevice));
        }

        void UploadTargets(const std::vector<float> &T) {
            CUDA_CHECK(cudaMemcpy(dT, T.data(), size_t(S) * layers[L] * sizeof(float), cudaMemcpyHostToDevice));
        }

        void UploadPopulation(const std::vector<float> &packed, bool resetMomentum = true) {
            CUDA_CHECK(cudaMemcpy(dPackW, packed.data(), packed.size() * sizeof(float), cudaMemcpyHostToDevice));
            if (resetMomentum) {
                CUDA_CHECK(cudaMemset(dMom, 0, P * WPG * sizeof(float)));  // 换了权重，旧动量速度作废
            }
        }

        // ===== [V3] 逐个体批量模式：N 个个体各自用"自己的 genome 权重 + 自己的输入行" =====
        // 用途：寻路模拟每帧把所有活体的前向合并成一次 GPU 调用（个体数即 batch 数）。
        // 流程：GatherIndividualWeights（卡上按 nnId 重排权重块）→ UploadFrameInputs → ForwardIndividuals
        uint maxN = 0;               // 单帧最大个体数
        int *dNnId = nullptr;        // [maxN] 各个体的 genome 索引
        float *dInputsN = nullptr;   // [maxN × I] 各个体的输入行
        float *dGatherW = nullptr;   // [maxN × WPG] 收集后的权重块

        void InitIndividualMode(uint maxN_) {
            maxN = maxN_;
            CUDA_CHECK(cudaMalloc(&dNnId, size_t(maxN) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&dInputsN, size_t(maxN) * layers[0] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dGatherW, size_t(maxN) * WPG * sizeof(float)));
        }

        void GatherIndividualWeights(const std::vector<int> &ids) {
            if (ids.size() > maxN) {
                std::cerr << "[CUDA] ids.size() " << ids.size() << " > maxN " << maxN << "\n";
                exit(1);
            }
            CUDA_CHECK(cudaMemcpy(dNnId, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
            size_t total = ids.size() * WPG;
            GatherRowsKernel<<<uint((total + 255) / 256), 256>>>(dPackW, dNnId, dGatherW, WPG, total);
            CUDA_CHECK(cudaGetLastError());
        }

        void UploadFrameInputs(const std::vector<float> &inputs) {  // N×I 行优先
            CUDA_CHECK(cudaMemcpy(dInputsN, inputs.data(), inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        // 逐个体前向：item i 用 dGatherW 第 i 块权重 + dInputsN 第 i 行，返回 [N×O]
        std::vector<float> ForwardIndividuals(uint N) {
            if (N > maxN) {
                N = maxN;
            }
            float alpha = 1.f, beta = 0.f;
            for (size_t l = 1; l <= L; ++l) {
                int in = int(layers[l - 1]), out = int(layers[l]);
                // [FIX] 本层输入是上一层激活 dAct[l-1]（曾误写 dAct[l]，读到未初始化内存）
                const float *Bsrc = (l == 1 ? dInputsN : dAct[l - 1]);
                CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                       out, 1, in, &alpha,
                                                       dGatherW + wOff[l], in, (long long) WPG,           // 各 item 自己的 W_l
                                                       Bsrc, in, (long long) in,                          // 各 item 自己的输入/上层激活行
                                                       &beta,
                                                       dAct[l], out, (long long) out,                     // 各 item 自己的输出行
                                                       int(N)));
                size_t total = size_t(N) * out;
                AddBiasActivateKernel<<<uint((total + 255) / 256), 256>>>(dAct[l], dGatherW, total, out,
                                                                          size_t(out), WPG, bOff[l]);
                CUDA_CHECK(cudaGetLastError());
            }
            std::vector<float> outputs(size_t(N) * layers[L]);
            CUDA_CHECK(cudaMemcpy(outputs.data(), dAct[L], outputs.size() * sizeof(float), cudaMemcpyDeviceToHost));
            return outputs;
        }

        // 批量前向：X [S×I] → 各层激活。batchCount=P；l=1 时输入共享（strideA=0）
        void Forward() {
            float alpha = 1.f, beta = 0.f;
            int batch = int(P);
            for (size_t l = 1; l <= L; ++l) {
                int in = int(layers[l - 1]), out = int(layers[l]);
                long long strideB = (long long) WPG;  // 每 genome 的 W_l
                CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                       out, int(S), in, &alpha,
                                                       dPackW + wOff[l], in, strideB,
                                                       (l == 1 ? dX : dAct[l - 1]), in, (l == 1 ? 0LL : (long long) S * in),
                                                       &beta,
                                                       dAct[l], out, (long long) S * out,
                                                       batch));
                size_t total = size_t(P) * S * out;
                uint blocks = uint((total + 255) / 256);
                // g = idx/(S*out)：population 模式下各 genome 取各自 bias，P=1 时 g 恒 0
                AddBiasActivateKernel<<<blocks, 256>>>(dAct[l], dPackW, total, out,
                                                       size_t(S) * out, WPG, bOff[l]);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // 单 genome（P=1）全批量训练一步：前向 + 反向 + 梯度均值更新
        // [V3] count：本批实际样本数（≤ 构造时的 S；0=全部）——classification 的样本数随用户点击变化
        void TrainEpoch(float lr, uint count = 0, float momentum = 0.f) {
            if (count == 0 || count > S) {
                count = S;
            }
            Forward();

            float alpha = 1.f, beta = 0.f;
            int outL = int(layers[L]);
            size_t total = size_t(count) * outL;
            OutputDeltaKernel<<<uint((total + 255) / 256), 256>>>(dAct[L], dT, dD, total);
            CUDA_CHECK(cudaGetLastError());

            float lrScale = lr / float(count);
            float *dCur = dD, *dPrev = dD + size_t(S) * maxLayer;  // [FIX] 双缓冲：前半=当前层 delta，后半=上一层写入区
            for (size_t l = L; l >= 1; --l) {
                int in = int(layers[l - 1]), out = int(layers[l]);
                const float *actPrev = (l == 1 ? dX : dAct[l - 1]);
                // gradW_l [out×in] = D_l^T × Act_{l-1}：cm 视角 C_cm[in×out] = Act_cm[in×S] × D_cm[out×S]^T
                // （D 的 cm 视图 ld=out < k=S，必须 OP_T，否则 ldb 非法）
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                         in, out, int(count), &alpha,
                                         actPrev, in,
                                         dCur, out,
                                         &beta,
                                         dGrad + wOff[l], in));
                // gradB_l = Σ_s D_l
                SumColsKernel<<<dim3(uint(out)), 256>>>(dCur, dGrad + bOff[l], int(count), out);
                CUDA_CHECK(cudaGetLastError());
                // 更新（clamp 截断，语义同 lib）
                ApplyUpdateKernel<<<uint((wNum[l] + 255) / 256), 256>>>(dPackW + wOff[l], dGrad + wOff[l], dMom + wOff[l], wNum[l], lrScale, momentum, Opts.WeightRange);
                ApplyUpdateKernel<<<uint((layers[l] + 255) / 256), 256>>>(dPackW + bOff[l], dGrad + bOff[l], dMom + bOff[l], layers[l], lrScale, momentum, Opts.BiasRange);
                CUDA_CHECK(cudaGetLastError());

                if (l > 1) {
                    // D_{l-1} = D_l × W_l：自定义核（消除 cuBLAS B/C 别名问题）
                    DeltaPropKernel<<<uint((size_t(count) * in + 255) / 256), 256>>>(dCur, dPackW + wOff[l], dPrev, int(count), in, out);
                    size_t nPrev = size_t(count) * in;
                    MulDerivKernel<<<uint((nPrev + 255) / 256), 256>>>(dPrev, dAct[l - 1], nPrev);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
        }

        // [CUDA] 回读打包权重（调试/验证用）
        std::vector<float> GetWeights() {
            std::vector<float> w(P * WPG);
            CUDA_CHECK(cudaMemcpy(w.data(), dPackW, w.size() * sizeof(float), cudaMemcpyDeviceToHost));
            return w;
        }

        // [V3] 区间小批量训练：只对行 [offset, offset+size) 做前向+反向+更新。
        // 乱序小批量恢复在线 SGD 的样本噪声——对称分类数据下全批量均值梯度会困在
        // "输出全 0.5" 的对称鞍点（classification CUDA 版 fitness 恒 0.5 的第二根因）
        void TrainBatch(float lr, float momentum, uint offset, uint size) {
            if (size == 0 || offset + size > S) {
                return;
            }
            float alpha = 1.f, beta = 0.f;
            // 前向（区间）
            for (size_t l = 1; l <= L; ++l) {
                int in = int(layers[l - 1]), out = int(layers[l]);
                const float *Bsrc = (l == 1 ? dX + offset * in : dAct[l - 1] + offset * in);
                CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                       out, int(size), in, &alpha,
                                                       dPackW + wOff[l], in, (long long) WPG,
                                                       Bsrc, in, (long long) in,
                                                       &beta,
                                                       dAct[l] + offset * out, out, (long long) out,
                                                       1));
                size_t total = size_t(size) * out;
                AddBiasActivateKernel<<<uint((total + 255) / 256), 256>>>(dAct[l] + offset * out, dPackW, total, out,
                                                                          size_t(out), WPG, bOff[l]);
                CUDA_CHECK(cudaGetLastError());
            }

            int outL = int(layers[L]);
            size_t total = size_t(size) * outL;
            OutputDeltaKernel<<<uint((total + 255) / 256), 256>>>(dAct[L] + offset * outL, dT + offset * outL, dD, total);
            CUDA_CHECK(cudaGetLastError());

            float lrScale = lr / float(size);
            float *dCur = dD, *dPrev = dD + size_t(S) * maxLayer;
            for (size_t l = L; l >= 1; --l) {
                int in = int(layers[l - 1]), out = int(layers[l]);
                const float *actPrev = (l == 1 ? dX + offset * in : dAct[l - 1] + offset * in);
                std::cerr << "[TB] l=" << l << " in=" << in << " out=" << out << " size=" << size << " off=" << offset << " layers:";
                for (auto v: layers) std::cerr << " " << v;
                std::cerr << " hides:";
                for (auto v: Opts.FCNN_hideLayers) std::cerr << " " << v;
                std::cerr << "\n";
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                         in, out, int(size), &alpha,
                                         actPrev, in,
                                         dCur, out,
                                         &beta,
                                         dGrad + wOff[l], in));
                SumColsKernel<<<dim3(uint(out)), 256>>>(dCur, dGrad + bOff[l], int(size), out);
                CUDA_CHECK(cudaGetLastError());
                ApplyUpdateKernel<<<uint((wNum[l] + 255) / 256), 256>>>(dPackW + wOff[l], dGrad + wOff[l], dMom + wOff[l], wNum[l], lrScale, momentum, Opts.WeightRange);
                ApplyUpdateKernel<<<uint((layers[l] + 255) / 256), 256>>>(dPackW + bOff[l], dGrad + bOff[l], dMom + bOff[l], layers[l], lrScale, momentum, Opts.BiasRange);
                CUDA_CHECK(cudaGetLastError());

                if (l > 1) {
                    DeltaPropKernel<<<uint((size_t(size) * in + 255) / 256), 256>>>(dCur, dPackW + wOff[l], dPrev, int(size), in, out);
                    size_t nPrev = size_t(size) * in;
                    MulDerivKernel<<<uint((nPrev + 255) / 256), 256>>>(dPrev, dAct[l - 1] + offset * in, nPrev);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
        }

        // [V3] 回传前 count 行输出（TrainEpoch 后 dAct[L] 仍是本批(更新前)输出，供每轮评估 fitness）
        std::vector<float> GetOutputs(uint count) {
            std::vector<float> outputs(size_t(count) * layers[L]);
            CUDA_CHECK(cudaMemcpy(outputs.data(), dAct[L], outputs.size() * sizeof(float), cudaMemcpyDeviceToHost));
            return outputs;
        }

        // 种群批量评估：返回 host 上的输出 [P][S][O]（行优先，每 genome 连续 S×O）
        std::vector<float> EvaluatePopulation() {
            Forward();
            std::vector<float> outputs(size_t(P) * S * layers[L]);
            CUDA_CHECK(cudaMemcpy(outputs.data(), dAct[L], outputs.size() * sizeof(float), cudaMemcpyDeviceToHost));
            return outputs;
        }
    };
}

#endif //MYNEAT_CUDA_NN_HPP
