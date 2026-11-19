#pragma once

#ifndef MYNEAT_OPTIONS_HPP
#define MYNEAT_OPTIONS_HPP

#include "Util.hpp"
#include <thread>
#include "ThreadPool.hpp"
#include <filesystem>
#include <shared_mutex>

namespace znn {
    struct Option {
        ulong InputSize = 0;
        ulong OutputSize = 0;
        uint PopulationSize = 50;
        uint IterationTimes = 500;  // 如果<=0则无限循环
        uint IterationCheckPoint = 0; // 如果<=0则不启用
        std::string CheckPointPath = std::filesystem::temp_directory_path().string() + "/check_point";
        uint ChampionKeepSize = 10;
        uint ChampionToNewSize = 30;
        uint KeepWorstSize = 1;
        uint NewSize = 1;
        uint KeepComplexSize = 1;
        std::function<float(float)> ActiveFunction = Sigmoid;
        std::function<float(float)> DerivativeFunction = DerivativeSigmoid;
        std::function<float(const std::vector<float> &, const std::vector<float> &)> PrecisionFunction = AbsoluteDeviation;  // [OPT] const引用：std::function按值签名会在每次调用时拷贝整个样本向量
        float FitnessThreshold = 0.95f; // 如果<=0则不启用
        float MutateWeightRate = 0.1f;
        int MutateWeightNearRange = 2;
        float MutateWeightDirectOrNear = 0.5f;
        float WeightRange = 10.f;
        float BiasRange = 30.f;
        float NewNNWeightRange = 1.f;
        float NewNNBiasRange = 3.f;
        float MutateBiasRate = 0.1f;
        float MutateBiasNearRange = 2;
        float MutateBiasDirectOrNear = 0.5f;
        float MutateAddNeuronRate = 0.1f;
        float MutateAddConnectionRate = 0.5f;
        float MutateEnableConnectionRate = 0.3f;
        float CrossoverRate = 0.99f;
        float LearnRate = 0.3f;
        // [OPT] BP动量系数，默认0=关闭（数学结果与旧版SGD一致）。实测：在未归一化输入+
        // sigmoid饱和+LR 0.1~0.3 的本项目典型配置下，0.9 动量会累积过冲导致不收敛
        // （iris 卡 0.717、XOR 卡 0.75）；输入先用 NormalizeData 归一化再配 0.5~0.9 才有效
        float BPMomentum = 0.f;
        uint BackLearnTimesPerBatch = 15;
        uint ThreadCount = std::thread::hardware_concurrency();
        bool SaveEveryTime = true;
        bool StartWithFCNN = false;
        bool FFCNNInsteadOfFCNN = false;
        std::vector<ulong> FCNN_hideLayers;

#ifndef NO_3DNN

        bool Enable3dNN = true;
        bool Enable3dCalc = false;
        float X_Interval3d = 1.f;
        float Zy_Interval3d = 1.f;
        bool Enable3dRandPos = true;
        bool Enable3dOutputOneLine = true;
        uint Update3dIntercalMs = 1000;
        int ScreenWidth = 960;
        int ScreenHeight = 720;

#endif
    };

    // [FIX] ODR问题：static 全局变量在每个编译单元各有一份副本（多TU链接时 Opts/mtx/tPool 互不相通），
    // C++17 起应使用 inline 变量，保证全程序唯一实例
    inline Option Opts;

    inline std::shared_mutex mtx;
    inline BS::thread_pool tPool(Opts.ThreadCount);
}

#endif //MYNEAT_OPTIONS_HPP
