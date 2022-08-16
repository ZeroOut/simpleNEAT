#pragma once

#ifndef MYNEAT_POPULATION_HPP
#define MYNEAT_POPULATION_HPP

#include "Generation.hpp"

namespace znn {
    std::vector<NetworkGenome> Population;

    void CreatePopulation() {
        Population.clear();
        for (uint i = 0; i < Opts.PopulationSize; ++i) {
            Population.push_back(NewNN());
        }
    }

    void CreatePopulationFC(std::vector<int> hideLayers) {
        Population.clear();
        for (uint i = 0; i < Opts.PopulationSize; ++i) {
            Population.push_back(NewFCNN(hideLayers));
        }
    }

    void CreatePopulationByGiving() {
        ImportInnovations(Opts.CheckPointPath);  // 要先导入innov
        Population.clear();
        auto nn = znn::ImportNN(Opts.CheckPointPath);
        for (uint i = 0; i < Opts.PopulationSize; ++i) {
            Population.push_back(nn);
        }
    }

    std::map<NetworkGenome *, float> CalculateFitnessByWanted(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> wantedOutputs) {  // 二位数组，第一维是实验次数，第二维输入和预期输出
        std::map<NetworkGenome *, float> populationFitness;

        if (inputs.size() != wantedOutputs.size() || wantedOutputs[0].size() != Opts.OutputSize || inputs[0].size() != Opts.InputSize) {
            std::cerr << "intput length: " << inputs[0].size() << " Opts.InputSize: " << Opts.InputSize << "\nwanted length: " << wantedOutputs[0].size() << " Opts.OutputSize: " << Opts.OutputSize
                      << "\ninputs times: " << inputs.size() << " wanted times " << wantedOutputs.size() << std::endl;
            exit(0);
        }

        std::vector<std::future<void>> thisFuture;  // 如果用这个线程池的push_task函数，后面需要wait_for_tasks()，会卡死

        for (auto &nn : Population) {
            thisFuture.push_back(tPool.submit([&]() {
                float fitness = 0.f;
                for (uint i = 0; i < inputs.size(); ++i) {
                    std::vector<float> thisOutput = FeedForwardPredict(&nn, inputs[i]);
                    fitness += StandardDeviation(thisOutput, wantedOutputs[i]);
                }
                mtx.lock();
                populationFitness[&nn] = fitness / float(inputs.size());
                mtx.unlock();
            }));
        }

        for (auto &f : thisFuture) {
            f.wait();
        }

        return populationFitness;
    }
}

#endif //MYNEAT_POPULATION_HPP
