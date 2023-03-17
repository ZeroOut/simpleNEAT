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

        void CreatePopulationFC(std::vector<ulong> &hideLayers);

        void CreatePopulationByGiving();

        std::map<NetworkGenome *, float> CalculateFitnessByWanted(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> wantedOutputs);
    };

    void Population::CreatePopulation() {
        NeuralNetworks.clear();
        for (uint i = 0; i < Opts.PopulationSize; ++i) {
            NeuralNetworks.push_back(generation.neuralNetwork.NewNN());
        }
    }

    void Population::CreatePopulationFC(std::vector<ulong> &hideLayers) {
        NeuralNetworks.clear();
        for (uint i = 0; i < Opts.PopulationSize; ++i) {
            NeuralNetworks.push_back(generation.neuralNetwork.NewFCNN(hideLayers));
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

    std::map<NetworkGenome *, float> Population::CalculateFitnessByWanted(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> wantedOutputs) {  // 二维数组，第一维是实验次数，第二维输入和预期输出
        std::map<NetworkGenome *, float> populationFitness;

        if (inputs.size() != wantedOutputs.size() || wantedOutputs[0].size() != Opts.OutputSize || inputs[0].size() != Opts.InputSize) {
            std::cerr << "intput length: " << inputs[0].size() << " Opts.InputSize: " << Opts.InputSize << "\nwanted length: " << wantedOutputs[0].size() << " Opts.OutputSize: " << Opts.OutputSize
                      << "\ninputs times: " << inputs.size() << " wanted times " << wantedOutputs.size() << std::endl;
            exit(0);
        }

        std::vector<std::future<void>> thisFuture;  // 如果用这个线程池的push_task函数，后面需要wait_for_tasks()，会卡死

        for (auto &nn: NeuralNetworks) {
            thisFuture.push_back(tPool.submit([&]() {
                float fitness = 0.f;
                for (uint i = 0; i < inputs.size(); ++i) {
                    std::vector<float> thisOutputs = generation.neuralNetwork.FeedForwardPredict(&nn, inputs[i], true);
                    fitness += GetPrecision(thisOutputs, wantedOutputs[i]);
                }
                mtx.lock();
                populationFitness[&nn] = fitness / float(inputs.size());
                mtx.unlock();
            }));
        }

        for (auto &f: thisFuture) {
            f.wait();
        }

        return populationFitness;
    }
}

#endif //MYNEAT_POPULATION_HPP
