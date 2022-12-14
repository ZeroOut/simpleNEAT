#pragma once

#ifndef MYNEAT_OPTIONS_HPP
#define MYNEAT_OPTIONS_HPP

#include "Util.hpp"
#include <thread>
#include "ThreadPool.hpp"
#include <filesystem>

namespace znn {
    struct Option {
        ulong InputSize = 0;
        ulong OutputSize = 0;
        uint PopulationSize = 100;
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
        float FitnessThreshold = 0.99f; // 如果<=0则不启用
        float MutateWeightRate = 0.1f;
        int MutateWeightNearRange = 2;
        float MutateWeightDirectOrNear = 0.5f;
        int WeightRange = 6;
        float MutateBiasRate = 0.1f;
        int MutateBiasNearRange = 2;
        float MutateBiasDirectOrNear = 0.5f;
        int BiasRange = 3;
        float MutateAddNeuronRate = 0.003f;
        float MutateAddConnectionRate = 0.5f;
        float MutateEnableConnectionRate = 0.3f;
        float CrossoverRate = 0.99f;
        float LearnRate = 0.3f;
        uint ThreadCount = std::thread::hardware_concurrency();

        bool Enable3dNN = false;
        float X_Interval3d = 1.f;
        float Zy_Interval3d = 1.f;
        bool Enable3dRandPos = true;
        uint Update3dIntercalMs = 1000;

    };

    static Option Opts;

    static std::mutex mtx;
    static BS::thread_pool tPool(Opts.ThreadCount);
}

#endif //MYNEAT_OPTIONS_HPP
