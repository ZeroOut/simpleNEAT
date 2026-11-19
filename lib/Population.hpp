#pragma once

#ifndef MYNEAT_POPULATION_HPP
#define MYNEAT_POPULATION_HPP

#include "Generation.hpp"

namespace znn {
    class Population {
    public:
        Generation generation;
        std::vector<NetworkGenome> NeuralNetworks;

        void CreatePopulation();

        void CreatePopulationByGiving();

        std::unordered_map<NetworkGenome *, float> CalculateFitnessByWanted(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &wantedOutputs);
    };

    void Population::CreatePopulation() {
        NeuralNetworks.clear();
        for (uint i = 0; i < Opts.PopulationSize; ++i) {
            if (Opts.StartWithFCNN) {
                NeuralNetworks.push_back(generation.neuralNetwork.NewFCNN());
            } else {
                NeuralNetworks.push_back(generation.neuralNetwork.NewNN());
            }
        }
    }

    void Population::CreatePopulationByGiving() {
        generation.neuralNetwork.ImportInnovations(Opts.CheckPointPath);  // 要先导入innov
        NeuralNetworks.clear();
        auto nn = generation.neuralNetwork.ImportNN(Opts.CheckPointPath);
        NeuralNetworks.push_back(nn);
        for (uint i = 1; i < Opts.PopulationSize; ++i) {
            auto nn0 = nn;
            generation.MutateNetworkGenome(nn0);
            NeuralNetworks.push_back(nn0);
        }
    }

    // [OPT] const 引用：每个样本×每个体×每代调用一次，按值传参的向量拷贝是纯开销
    float GetPrecision(const std::vector<float> &outputs, const std::vector<float> &wantedOutputs) {
        return 1.f - Opts.PrecisionFunction(outputs, wantedOutputs);
    }

    std::unordered_map<NetworkGenome *, float> Population::CalculateFitnessByWanted(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &wantedOutputs) {  // 二维数组，第一维是实验次数，第二维输入和预期输出。 [OPT] const引用：避免整份数据集拷贝
        std::unordered_map<NetworkGenome *, float> populationFitness;

        // [FIX] 原代码 inputs 为空时 wantedOutputs[0] 先越界再触发后面的长度检查；先判空再比对
        if (inputs.empty() || wantedOutputs.empty() ||
            inputs.size() != wantedOutputs.size() || wantedOutputs[0].size() != Opts.OutputSize || inputs[0].size() != Opts.InputSize) {
            // [FIX] 错误打印本身也要判空，否则空 inputs 在报错路径先越界崩溃
            std::cerr << "intput length: " << (inputs.empty() ? 0 : inputs[0].size()) << " Opts.InputSize: " << Opts.InputSize
                      << "\nwanted length: " << (wantedOutputs.empty() ? 0 : wantedOutputs[0].size()) << " Opts.OutputSize: " << Opts.OutputSize
                      << "\ninputs times: " << inputs.size() << " wanted times " << wantedOutputs.size() << std::endl;
            exit(0);
        }

        // [FIX] 此处 lambda 按引用捕获 nn（&nn 是循环变量引用），任务可能在循环推进后执行；
        // 实际上 for-range 的 nn 每次迭代是独立引用绑定当前元素，且 tPool.wait_for_tasks() 在 return 前完成，
        // 是安全的，但为消除对执行时序的依赖，改为按值捕获当前元素地址
        for (auto &nn: NeuralNetworks) {
            NetworkGenome *nnPtr = &nn;  // [FIX] 先取地址再入任务，避免 lambda 引用捕获循环变量
            tPool.push_task([this, nnPtr, &inputs, &wantedOutputs, &populationFitness]() {
                float fitness = 0.f;
                for (uint i = 0; i < inputs.size(); ++i) {
                    std::vector<float> thisOutputs = generation.neuralNetwork.FeedForwardPredict(nnPtr, inputs[i], false);
                    fitness += GetPrecision(thisOutputs, wantedOutputs[i]);
                }
                fitness /= float(inputs.size());

                mtx.lock();
                populationFitness[nnPtr] = fitness;
                mtx.unlock();
            });
        }

        tPool.wait_for_tasks();

        return populationFitness;
    }
}

#endif //MYNEAT_POPULATION_HPP
