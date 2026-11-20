#pragma once

#ifndef MYNEAT_CUDA_NN_HPP
#define MYNEAT_CUDA_NN_HPP

// [CUDA] 基于 cuBLAS 的全连接网络批量引擎（原文件是 FCNNFeedForwardPredict 的 CPU 草稿，无 CUDA 代码，已重写）
//
// ===== 编译说明 =====
// 本文件是头文件，被 *_cuda.cpp 通过 #include 使用；含 __global__ 内核，所以**包含它的 .cpp
// 必须用 nvcc 编译**（-x cu 强制按 CUDA 源处理），链接 cuBLAS：
//
//   无渲染测试（iris/xor/基准）：
//     nvcc -O3 -std=c++20 -arch=native test_iris_FC_cuda.cpp -x cu \
//          -o build/test_iris_FC_cuda -lcublas -lraylib
//     （-lraylib 是 lib 头文件的链接依赖，即使 Enable3dNN=false 也要）
//
//   SFML 渲染版（classification）：
//     nvcc -O3 -std=c++20 -arch=native test_classification_sfml_cuda.cpp -x cu \
//          -o build/test_classification_sfml_cuda -lcublas -lsfml-graphics -lsfml-window -lsfml-system -lraylib
//
//   raylib 渲染版（pathfinder）：
//     nvcc -O3 -std=c++20 -arch=native test_pathfindder_raylib_cuda.cpp -x cu \
//          -o build/test_pathfindder_raylib_cuda -lcublas -lraylib
//
//   语法快速检查（不产出二进制）：
//     nvcc -std=c++20 -arch=native -DNO_3DNN -I. -fsyntax-only <任意包含本文件的 .cpp>
//
//   CMake：-DENABLE_CUDA=ON（test_bench_fcnn / test_bench_ga / 两个 *_cuda 渲染版均已配置）
//   -arch=native 自动匹配本机 GPU（如 RTX 2080 Ti = sm_75）；设备端激活函数固定 Sigmoid。
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
#include <set>      // CudaNeatEngine 的层值集合
#include <map>      // (to,from) 分组
#include <unordered_map>
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
                    // [FIX] 交替双缓冲：上一层 delta 已在 dPrev，下一轮必须读它——
                    // 缺此交换时 gradW 永远读旧层 delta（iris/xor 退化 0.80/0.5 的根因）
                    std::swap(dCur, dPrev);
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
                    std::swap(dCur, dPrev);  // [FIX] 同上
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

    // [CUDA] supernet 评估用核函数（namespace 作用域——__global__ 不能是成员函数）
    __global__ void NeatCopyInputKernel(const float *X, float *act, size_t total, int in, size_t S,
                                        size_t sampleStride, size_t genomeStride, size_t inSlot) {
        size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        size_t g = idx / (S * size_t(in));
        size_t r = idx % (S * size_t(in));
        size_t s = r / size_t(in);
        size_t i = r % size_t(in);
        // [FIX] act 样本步距是 nNeurons（曾线性写 r=s*I+i，样本 1 的输入踩进样本 0 的
        // 隐藏层槽位 → 全场错乱）
        act[g * genomeStride + s * sampleStride + inSlot + i] = X[r];
    }

    // [FIX] 跨步寻址：act 布局 [P × S × nNeurons]（曾线性写 A[idx]，sigmoid 落到错误槽位、
    // 真实槽位保持原始 GEMM 值 → 输出 >1 / 全 0）
    __global__ void NeatMulMaskKernel(float *act, const float *mask, size_t total, int Nl, size_t S,
                                      size_t sampleStride, size_t genomeStride, size_t layerOff, size_t nNeurons) {
        size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        size_t g = idx / (S * size_t(Nl));
        size_t r = idx % (S * size_t(Nl));
        size_t sIdx = r / size_t(Nl);
        size_t j = r % size_t(Nl);
        size_t addr = g * genomeStride + sIdx * sampleStride + layerOff + j;
        act[addr] *= mask[g * nNeurons + layerOff + j];
    }

    // 跨步清零：memset 是线性写，会跨过样本/基因组边界清掉其它层的槽位（如 g0s1 的输入）
    __global__ void NeatZeroLayerKernel(float *act, size_t total, int Nl, size_t S,
                                        size_t sampleStride, size_t genomeStride, size_t layerOff) {
        size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        size_t g = idx / (S * size_t(Nl));
        size_t r = idx % (S * size_t(Nl));
        size_t sIdx = r / size_t(Nl);
        size_t j = r % size_t(Nl);
        act[g * genomeStride + sIdx * sampleStride + layerOff + j] = 0.f;
    }

    // [FIX] 输出收集：输出槽位在 act 里按样本跨步（间隔 nNeurons），2D memcpy 无法表达
    // 两级跨步（曾把 s≥1 的输入槽当输出读回 → 恒 1.0）
    __global__ void NeatGatherOutputsKernel(const float *act, float *outs, size_t total, int O, size_t S,
                                            size_t sampleStride, size_t genomeStride, size_t outOff) {
        size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        size_t g = idx / (S * size_t(O));
        size_t r = idx % (S * size_t(O));
        size_t sIdx = r / size_t(O);
        size_t j = r % size_t(O);
        outs[idx] = act[g * genomeStride + sIdx * sampleStride + outOff + j];
    }

    // 跨步 bias+sigmoid：同上布局
    __global__ void NeatBiasActivateKernel(float *act, const float *bias, size_t total, int Nl, size_t S,
                                           size_t sampleStride, size_t genomeStride, size_t layerOff, size_t nNeurons) {
        size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        size_t g = idx / (S * size_t(Nl));
        size_t r = idx % (S * size_t(Nl));
        size_t sIdx = r / size_t(Nl);
        size_t j = r % size_t(Nl);
        size_t addr = g * genomeStride + sIdx * sampleStride + layerOff + j;
        act[addr] = 1.f / (1.f + expf(-(act[addr] + bias[g * nNeurons + layerOff + j])));
    }

    // [CUDA] NEAT supernet 评估引擎（评估 only，无 BP）
    // 思路（GPU-NEAT 主流方案）：把本代全部个体的神经元/连接取并集构成 supernet——
    // 每个个体 = supernet 的一组权重（不存在的连接 w=0，数学上等效禁用）+ 神经元存在 mask
    // （不存在的神经元激活强制 0，防止 sigmoid(0)=0.5 污染下游）。supernet 层结构对全体统一
    // （NEAT 的 innovation 机制保证同 Id 神经元同 Layer），因此可以逐层 strided-batched GEMM 批量。
    // 代价：计算量 = supernet 大小而非每个体实际大小（个体拓扑差异大时浪费增多，但稠密 GEMM 的
    // GPU 吞吐通常仍远超 CPU 逐个前向）。
    struct CudaNeatEngine {
        // supernet 描述
        std::vector<double> layers;                       // 排序去重的层值
        std::vector<std::vector<ulong>> layerNeuronIds;   // 每层内神经元 Id（按 Id 排序）
        std::unordered_map<ulong, std::pair<int, int>> neuronSlot;  // Id -> (层号, 层内序号)
        std::vector<size_t> neuronOff;                    // 每层神经元槽在打包块内的偏移
        size_t nNeurons = 0;                              // supernet 神经元总数
        std::vector<std::array<ulong, 2>> conns;          // supernet 连接（仅 Enable 连接的并集）
        std::unordered_map<uint64_t, int> connSlot;       // (from<<32|to) -> 连接序号
        struct ConnGroup {
            int fromLayer;
            size_t off;    // 每 genome 权重块内的偏移
            int Nl, Np;    // 稠密矩阵维度 [Nl × Np]
        };
        std::vector<std::vector<ConnGroup>> connGroups;   // 每 to 层的 (源层) 分组——NEAT 连接可跨层
                                                          // 直连（输入→输出绕过隐藏层），必须按 (to,from)
                                                          // 二元组分组累加，只算相邻层会丢直连边。
                                                          // 权重按组稠密打包 [Nl × Np]（缺失连接=0，
                                                          // 稀疏列表会与 GEMM 的稠密 A 操作数错位）
        size_t wTotal = 0;                                // 每 genome 权重块总宽
        uint P = 0, S = 0, I = 0, O = 0;

        float *dW = nullptr;      // [P × nConns] 每体权重（不存在=0），按 to 层分组连续
        float *dBias = nullptr;   // [P × nNeurons] 每体偏置（不存在=0），按层偏移
        float *dMask = nullptr;   // [P × nNeurons] 神经元存在掩码
        float *dXn = nullptr;     // [S × I]
        float *dAct = nullptr;    // [P × S × nNeurons] 全部激活（按层偏移寻址）
        float *dOuts = nullptr;   // [P × S × O] 收集后的连续输出
        cublasHandle_t handle = nullptr;
        bool built = false;
        size_t capW = 0, capN = 0, capA = 0;

        CudaNeatEngine(uint P_, uint S_) : P(P_), S(S_) {
            CUBLAS_CHECK(cublasCreate(&handle));
        }

        ~CudaNeatEngine() {
            if (handle) {
                cublasDestroy(handle);
            }
            cudaFree(dW);
            cudaFree(dBias);
            cudaFree(dMask);
            cudaFree(dXn);
            cudaFree(dAct);
            cudaFree(dOuts);
        }

        // 每代重建 supernet（个体拓扑会变），并把种群权重/偏置/掩码打包上传
        void BuildSupernet(const std::vector<NetworkGenome> &pop, uint inputSize, uint outputSize) {
            I = inputSize;
            O = outputSize;
            layers.clear();
            layerNeuronIds.clear();
            neuronSlot.clear();
            conns.clear();
            connSlot.clear();
            std::set<double> layerSet;
            std::map<ulong, double> id2Layer;

            for (auto &g: pop) {
                for (auto &n: g.Neurons) {
                    auto it = id2Layer.find(n.Id);
                    if (it == id2Layer.end()) {
                        id2Layer[n.Id] = n.Layer;
                        layerSet.insert(n.Layer);
                    } else if (it->second != n.Layer) {
                        std::cerr << "[CudaNeat] neuron " << n.Id << " has inconsistent layer across population\n";
                        exit(1);
                    }
                }
            }
            layers.assign(layerSet.begin(), layerSet.end());
            layerNeuronIds.resize(layers.size());
            neuronOff.assign(layers.size() + 1, 0);

            // 每层神经元按 Id 排序；保证同 Id 跨个体同槽
            for (auto &kv: id2Layer) {
                int li = int(std::find(layers.begin(), layers.end(), kv.second) - layers.begin());
                layerNeuronIds[li].push_back(kv.first);
            }
            for (auto &v: layerNeuronIds) {
                std::sort(v.begin(), v.end());
            }
            for (size_t l = 0; l < layers.size(); ++l) {
                neuronOff[l + 1] = neuronOff[l] + layerNeuronIds[l].size();
                for (size_t j = 0; j < layerNeuronIds[l].size(); ++j) {
                    neuronSlot[layerNeuronIds[l][j]] = {int(l), int(j)};
                }
            }
            nNeurons = neuronOff[layers.size()];

            // 连接并集（只收 Enable；同 (from,to) 槽位共享，禁用个体 w=0）
            for (auto &g: pop) {
                for (auto &c: g.Connections) {
                    if (!c.Enable) {
                        continue;
                    }
                    uint64_t key = (uint64_t(c.ConnectedNeuronId[0]) << 32) | c.ConnectedNeuronId[1];
                    if (connSlot.find(key) == connSlot.end()) {
                        connSlot[key] = int(conns.size());  // 记录 conns 下标（分组后经 finalSlot 映射到打包槽位）
                        conns.push_back(c.ConnectedNeuronId);
                    }
                }
            }

            // 连接按 (to 层, from 层) 二元组分组——跨层直连边归入 (to, from<to) 组
            std::map<int, std::map<int, std::vector<int>>> byPair;
            for (int ci = 0; ci < int(conns.size()); ++ci) {
                int to = neuronSlot[conns[ci][1]].first;
                int from = neuronSlot[conns[ci][0]].first;
                if (from >= to) {
                    std::cerr << "[CudaNeat] conn " << conns[ci][0] << "->" << conns[ci][1]
                              << " has from-layer >= to-layer\n";
                    exit(1);
                }
                byPair[to][from].push_back(ci);
            }
            connGroups.assign(layers.size(), {});
            std::vector<int> finalSlot(conns.size());
            size_t wOff = 0;
            for (size_t l = 0; l < layers.size(); ++l) {
                int Nl = int(layerNeuronIds[l].size());
                for (auto &kv: byPair[int(l)]) {
                    int Np = int(layerNeuronIds[kv.first].size());
                    ConnGroup grp;
                    grp.fromLayer = kv.first;
                    grp.off = wOff;
                    grp.Nl = Nl;
                    grp.Np = Np;
                    // 稠密 [Nl × Np]（rm）：元素 (输出 j, 输入 i) 在 j*Np + i
                    for (int ci: kv.second) {
                        int j = neuronSlot[conns[ci][1]].second;
                        int i = neuronSlot[conns[ci][0]].second;
                        finalSlot[ci] = int(wOff + size_t(j) * Np + i);
                    }
                    wOff += size_t(Nl) * Np;
                    connGroups[l].push_back(grp);
                }
            }
            wTotal = wOff;

            // 打包 [P × wTotal]
            std::vector<float> wPack(size_t(P) * wTotal, 0.f);
            std::vector<float> bPack(size_t(P) * nNeurons, 0.f);
            std::vector<float> mPack(size_t(P) * nNeurons, 0.f);
            for (uint p = 0; p < pop.size(); ++p) {
                auto &g = pop[p];
                for (auto &n: g.Neurons) {
                    auto sl = neuronSlot[n.Id];
                    size_t off = neuronOff[sl.first] + sl.second;
                    bPack[size_t(p) * nNeurons + off] = n.Bias;
                    mPack[size_t(p) * nNeurons + off] = 1.f;
                }
                for (auto &c: g.Connections) {
                    if (!c.Enable) {
                        continue;
                    }
                    uint64_t key = (uint64_t(c.ConnectedNeuronId[0]) << 32) | c.ConnectedNeuronId[1];
                    wPack[size_t(p) * wTotal + finalSlot[connSlot[key]]] = c.Weight;
                }
            }

            // [FIX] supernet 每代可能增长（结构变异加神经元），按需重分配
            size_t needW = size_t(P) * wTotal, needN = size_t(P) * nNeurons, needA = size_t(P) * S * nNeurons;
            if (!built || dW == nullptr || capW < needW || capN < needN || capA < needA) {
                cudaFree(dW);
                cudaFree(dBias);
                cudaFree(dMask);
                cudaFree(dAct);
                cudaFree(dXn);
                CUDA_CHECK(cudaMalloc(&dW, std::max(needW, size_t(1)) * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&dBias, std::max(needN, size_t(1)) * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&dMask, std::max(needN, size_t(1)) * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&dAct, std::max(needA, size_t(1)) * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&dOuts, std::max(size_t(P) * S * O, size_t(1)) * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&dXn, std::max(size_t(S) * I, size_t(1)) * sizeof(float)));
                capW = needW;
                capN = needN;
                capA = needA;
                built = true;
            }
            CUDA_CHECK(cudaMemcpy(dW, wPack.data(), wPack.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dBias, bPack.data(), bPack.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dMask, mPack.data(), mPack.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        void UploadX(const std::vector<float> &X) {  // [S × I]
            CUDA_CHECK(cudaMemcpy(dXn, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        // 批量评估：返回 [P × S × O]（行优先，每 genome 连续 S×O）
        std::vector<float> Evaluate() {
            float alpha = 1.f;
            size_t inSlot = neuronOff[0];  // 输入层槽偏移（Layer 0 必为第 0 层）
            // 输入层：复制 X（P 份共享同一输入）
            {
                size_t total = size_t(P) * S * I;
                NeatCopyInputKernel<<<uint((total + 255) / 256), 256>>>(dXn, dAct, total, int(I), S,
                                                                        nNeurons, size_t(S) * nNeurons, inSlot);
                CUDA_CHECK(cudaGetLastError());
            }

            // 逐 supernet 层前向（跳过输入层）。NEAT 连接可跨层直连——每个 to 层对每个
            // 有入边的源层做一次 GEMM 累加（beta 首组 0、后续 1），再 bias+sigmoid+mask。
            // cm 矩阵的 ld 全部用 supernet 宽 nNeurons（act 里跨样本步距）
            for (size_t l = 1; l < layers.size(); ++l) {
                int Nl = int(layerNeuronIds[l].size());
                float *actCur = dAct + neuronOff[l];
                size_t total = size_t(P) * S * Nl;
                // [FIX] 先清零再累加：存在但全部入边被禁用的槽位没有 GEMM 组覆盖，
                // 不清零会读未初始化显存 → 垃圾传播；清零后 "无入边 → sigmoid(0+bias)"
                // 与 CPU 语义精确一致。memset 是线性写会殃及相邻槽位，必须用跨步核
                NeatZeroLayerKernel<<<uint((total + 255) / 256), 256>>>(dAct, total, Nl, S,
                                                                       nNeurons, size_t(S) * nNeurons, neuronOff[l]);
                CUDA_CHECK(cudaGetLastError());
                for (auto &grp: connGroups[l]) {
                    int Np = int(layerNeuronIds[grp.fromLayer].size());
                    const float *actPrev = dAct + neuronOff[grp.fromLayer];
                    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                           Nl, int(S), Np, &alpha,
                                                           dW + grp.off, Np, (long long) wTotal,   // 每 genome 权重块
                                                           actPrev, (long long) nNeurons, (long long) S * nNeurons,
                                                           &alpha,
                                                           actCur, (long long) nNeurons, (long long) S * nNeurons,
                                                           int(P)));
                }
                // bias + sigmoid + 存在掩码（无入边组 = 孤立槽：sigmoid(bias) 后被 mask 修正）
                // [FIX] 传 dAct 完整基址（actCur 已含 layerOff，内核内再加一次 = 双重偏移，
                // sigmoid 写到更后面的槽位、真实槽位保持原始值）
                NeatBiasActivateKernel<<<uint((total + 255) / 256), 256>>>(dAct, dBias, total, Nl, S,
                                                                           nNeurons, size_t(S) * nNeurons, neuronOff[l], nNeurons);
                CUDA_CHECK(cudaGetLastError());
                NeatMulMaskKernel<<<uint((total + 255) / 256), 256>>>(dAct, dMask, total, Nl, S,
                                                                      nNeurons, size_t(S) * nNeurons, neuronOff[l], nNeurons);
                CUDA_CHECK(cudaGetLastError());
            }

            // 回读输出层（Layer 1 = 最后一层；NEAT 输出神经元 Id = I..I+O-1 排序后天然对齐）。
            // [FIX] 输出槽位按样本跨步，先用收集核打包成连续 [P × S × O] 再一次拷回
            size_t total = size_t(P) * S * O;
            std::vector<float> outs(total);
            NeatGatherOutputsKernel<<<uint((total + 255) / 256), 256>>>(dAct, dOuts, total, int(O), S,
                                                                        nNeurons, size_t(S) * nNeurons, neuronOff[layers.size() - 1]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(outs.data(), dOuts, total * sizeof(float), cudaMemcpyDeviceToHost));
            return outs;
        }

        // [DEBUG] 回读全部激活 [P × S × nNeurons]
        std::vector<float> DebugAct() {
            std::vector<float> a(size_t(P) * S * nNeurons);
            CUDA_CHECK(cudaMemcpy(a.data(), dAct, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
            return a;
        }

        // [DEBUG] 回读打包权重 [P × wTotal]
        std::vector<float> DebugW() {
            std::vector<float> w(size_t(P) * wTotal);
            CUDA_CHECK(cudaMemcpy(w.data(), dW, w.size() * sizeof(float), cudaMemcpyDeviceToHost));
            return w;
        }
    };
}

#endif //MYNEAT_CUDA_NN_HPP
