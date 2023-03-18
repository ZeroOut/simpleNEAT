#pragma once

#ifndef MYNEAT_SIMPLENEAT_HPP
#define MYNEAT_SIMPLENEAT_HPP

//#define USING_RAYLIB

#include "Population.hpp"
#include <unistd.h>

namespace znn {
    struct BestOne {
        uint Gen = 0;
        NetworkGenome NN;
        float Fit;
    };

    class SimpleNeat {
    public:
        Population population;

        void StartNew();

        void StartWithCheckPoint();

        void Start();

        znn::BestOne TrainByWanted(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &wantedOutputs, const uint randomSize, const std::function<bool()> &isBreakFunc);

        std::vector<NetworkGenome *> OrderByFitness(std::map<NetworkGenome *, float> &M);

        std::vector<NetworkGenome *> OrderByComplex();

        BestOne TrainByInteractive(const std::function<std::map<NetworkGenome *, float>()> &interactiveFunc, const std::function<bool()> &isBreakFunc);
    };


    void CheckOptions() {
        if (Opts.ChampionKeepSize < 2) {
            std::cerr << "Opts.ChampionKeepSize at least 2, for crossing over" << std::endl;
            exit(0);
        }

        if (Opts.ChampionToNewSize + Opts.KeepWorstSize + Opts.NewSize + Opts.KeepComplexSize > Opts.PopulationSize) {
            std::cerr << "Opts.ChampionToNewSize + Opts.KeepWorstSize + Opts.NewSize + Opts.KeepComplexSize > Opts.PopulationSize" << std::endl;
            exit(0);
        }

        if (Opts.ChampionKeepSize * 3 > Opts.ChampionToNewSize) {
            std::cerr << "Opts.ChampionKeepSize * 3 > Opts.ChampionToNewSize" << std::endl;
            exit(0);
        }

        if (Opts.FitnessThreshold <= 0 && Opts.IterationTimes <= 0) {
            std::clog << "Warning: Opts.FitnessThreshold <= 0, Opts.IterationTimes <= 0, will be infinity loop" << std::endl;
        }

        tPool.reset(Opts.ThreadCount);
        srandom((unsigned) clock());

        if (Opts.Enable3dNN) {
            std::thread show3d([](){
                canClose3dNN = false;
                Show3dNN();
            });
            show3d.detach();
        }

        if (Opts.usingFCNN && Opts.FCNN_hideLayers.empty()) {
            std::cerr << "Opts.usingFCNN = " << Opts.usingFCNN << " and Opts.FCNN_hideLayers.size() = " << Opts.FCNN_hideLayers.size() << std::endl;
            exit(0);
        }
    }

    void SimpleNeat::StartNew() {
        CheckOptions();
        population.CreatePopulation();
    }

    void SimpleNeat::StartWithCheckPoint() {
        CheckOptions();
        population.CreatePopulationByGiving();
    }

    void SimpleNeat::Start() {
        if (access((Opts.CheckPointPath + ".innov").c_str(), F_OK) != -1 && access((Opts.CheckPointPath + ".nn").c_str(), F_OK) != -1) {
            StartWithCheckPoint();
        } else {
            std::clog << "Check point files are not exist, start new.\n";
            StartNew();
        }
    }

    bool cmpf(std::pair<NetworkGenome *, float> &a, std::pair<NetworkGenome *, float> &b) {
        return a.second > b.second;// 从大到小排列
    }

    std::vector<NetworkGenome *> SimpleNeat::OrderByFitness(std::map<NetworkGenome *, float> &M) {  // Comparator function to sort pairs according to second value
        std::vector<NetworkGenome *> result;
        std::vector<std::pair<NetworkGenome *, float> > A;// Declare vector of pairs
        for (auto &it: M) {  // Copy key-value pair from Map to vector of pairs
            A.push_back(it);
        }
        std::sort(A.begin(), A.end(), cmpf);// Sort using comparator function
        for (auto &it: A) {
            result.push_back(it.first);
        }
        return result;
    }

    bool cmpc(std::pair<NetworkGenome *, ulong> &a, std::pair<NetworkGenome *, ulong> &b) {
        return a.second > b.second;// 从大到小排列
    }

    std::vector<NetworkGenome *> SimpleNeat::OrderByComplex() {  // Comparator function to sort pairs according to second value
        std::vector<NetworkGenome *> result;
        std::vector<std::pair<NetworkGenome *, ulong> > A;// Declare vector of pairs
        for (auto &it: population.NeuralNetworks) {  // Copy key-value pair from Map to vector of pairs
            A.push_back(std::pair{&it, it.Connections.size()});
        }
        std::sort(A.begin(), A.end(), cmpc);// Sort using comparator function
        for (auto &it: A) {
            result.push_back(it.first);
        }
        return result;
    }

    BestOne SimpleNeat::TrainByWanted(const std::vector<std::vector<float>> &rawInputs, const std::vector<std::vector<float>> &rawWantedOutputs, const uint randomSize, const std::function<bool()> &isBreakFunc) {
        if (rawInputs.size() != rawWantedOutputs.size()) {
            std::cerr << "rawInputs size: " << rawInputs.size() << " != rawWantedOutputs size: " << rawWantedOutputs.size() << "\n";
            exit(0);
        }

        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<float>> wantedOutputs;

        if (randomSize > 0) {
            uint *randIdsPtr = GetRandIndex(rawInputs.size());

            for (uint i = 0; i < randomSize; ++i) {
                inputs.push_back(rawInputs[randIdsPtr[i]]);
                wantedOutputs.push_back(rawWantedOutputs[randIdsPtr[i]]);
            }

            delete[] randIdsPtr;
        } else {
            inputs = rawInputs;
            wantedOutputs = rawWantedOutputs;
        }

        auto populationFitness = population.CalculateFitnessByWanted(inputs, wantedOutputs);
        auto orderedPopulation = OrderByFitness(populationFitness);
        auto orderedByComplex = OrderByComplex();

        uint rounds = 1;
        float lastFitness = 0.f;

        for (; rounds <= Opts.IterationTimes || Opts.IterationTimes <= 0; ++rounds) {
            if (populationFitness[orderedPopulation[0]] > lastFitness || (Opts.IterationCheckPoint > 0 && rounds % Opts.IterationCheckPoint == 0)) {
                lastFitness = populationFitness[orderedPopulation[0]];
                std::cout << "gen: " << rounds << " ptr: " << orderedPopulation[0] << " age: " << orderedPopulation[0]->Age << " neurons: " << orderedPopulation[0]->Neurons.size() << " connections: "
                          << orderedPopulation[0]->Connections.size() << " fitness: " << populationFitness[orderedPopulation[0]] << "\n";
            }

            for (auto nn: orderedPopulation) {
                ++nn->Age;
            }

            if (Opts.FitnessThreshold > 0 && populationFitness[orderedPopulation[0]] >= Opts.FitnessThreshold || isBreakFunc()) {
                auto simplifiedBestNN = population.generation.neuralNetwork.SimplifyRemoveDisable(*orderedPopulation[0]);
                auto compressedLeftBestNN = population.generation.neuralNetwork.SimplifyRemoveUselessConnectionLeft(simplifiedBestNN);
                auto compressedRightBestNN = population.generation.neuralNetwork.SimplifyRemoveUselessConnectionRight(compressedLeftBestNN);

                if (Opts.IterationCheckPoint > 0) {
                    population.generation.neuralNetwork.ExportInnovations(Opts.CheckPointPath);
                    population.generation.neuralNetwork.ExportNN(simplifiedBestNN, Opts.CheckPointPath); // 导出导入的格式定为没有已禁用连接，断点不需要简化孤立连接
                }

                population.generation.neuralNetwork.ExportNN(compressedRightBestNN, "./champion");
                population.generation.neuralNetwork.ExportNNToDot(compressedRightBestNN, "./champion");

                if (Opts.Enable3dNN) {
                    tPool.push_task(Update3dNN, compressedRightBestNN, false);
                    std::cout << "需保持主线程不退出,防止3d显示bug\n";
                }

                return BestOne{.Gen = rounds, .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                        //                        .NN = *orderedPopulation[0],
                        .Fit = populationFitness[orderedPopulation[0]],};
            }

            if (Opts.IterationCheckPoint > 0 && rounds % Opts.IterationCheckPoint == 0) {
                auto simplifiedBestNN = population.generation.neuralNetwork.SimplifyRemoveDisable(*orderedPopulation[0]);

                if (Opts.IterationCheckPoint > 0) {
                    population.generation.neuralNetwork.ExportInnovations(Opts.CheckPointPath);
                    population.generation.neuralNetwork.ExportNN(simplifiedBestNN, Opts.CheckPointPath); // 导出导入的格式定为没有已禁用连接，断点不需要简化孤立连接
                }
            }

            std::vector<NetworkGenome> tmpPopulation(Opts.PopulationSize);
            std::vector<std::future<void>> thisFuture;// 如果用这个线程池的push_task函数，后面需要wait_for_tasks()，会卡死

            uint indexOutside = 0;
            for (auto &n: tmpPopulation) {
                thisFuture.push_back(tPool.submit([&](uint index, NetworkGenome *nn) {
                    if (index < Opts.ChampionToNewSize) {
                        *nn = *orderedPopulation[index % Opts.ChampionKeepSize];  // 选取ChampionKeepSize个个体填满前ChampionToNewSize个
                        if (index >= Opts.ChampionKeepSize && index < Opts.ChampionKeepSize * 2) {
                            population.generation.MutateNetworkGenome(*nn);  // 冠军一份副本进行变异
                        }
                        if (index >= Opts.ChampionKeepSize * 2) {
                            for (uint i = 0; i < inputs.size(); ++i) {  // 保留的冠军一份副本全部进行反向传播更新weight和bias
                                population.generation.BackPropagation(nn, inputs[i], wantedOutputs[i]);
                            }
                        }
                    } else if (index < Opts.PopulationSize - Opts.NewSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        long chooseAnother = random() % Opts.PopulationSize;
                        auto nn0 = orderedPopulation[random() % Opts.ChampionKeepSize];
                        auto nn1 = orderedPopulation[chooseAnother];
//                        auto nn1 = orderedPopulation[Opts.ChampionKeepSize + random() % (Opts.PopulationSize - Opts.ChampionKeepSize)];
                        *nn = population.generation.GetChildByCrossing(nn0, nn1);
                        if (random() % 2 == 0 || nn0 == nn1 || chooseAnother < Opts.ChampionKeepSize) {
                            population.generation.MutateNetworkGenome(*nn);  // 繁殖以后进行变异
                        }
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        if (Opts.usingFCNN) {
                            *nn = population.generation.neuralNetwork.NewFCNN();
                        } else {
                            *nn = population.generation.neuralNetwork.NewNN();
                        }
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize) {
                        *nn = *orderedByComplex[index % Opts.KeepComplexSize];
                        population.generation.EnableAllConnections(*nn);
                    } else {
                        *nn = *orderedPopulation[index];
                        population.generation.MutateNetworkGenome(*nn);
                    }
                }, indexOutside, &n));
                ++indexOutside;
            }

            for (auto &f: thisFuture) {
                f.wait();
            }

            if (Opts.Enable3dNN) {
                tPool.push_task(Update3dNN, *orderedPopulation[0], false);
            }

            population.NeuralNetworks = tmpPopulation;

            populationFitness.clear();
            orderedPopulation.clear();
            orderedByComplex.clear();

            if (randomSize > 0) {
                inputs.clear();
                wantedOutputs.clear();

                uint *randIdsPtr = GetRandIndex(rawInputs.size());

                for (uint i = 0; i < randomSize; ++i) {
                    inputs.push_back(rawInputs[randIdsPtr[i]]);
                    wantedOutputs.push_back(rawWantedOutputs[randIdsPtr[i]]);
                }

                delete[] randIdsPtr;
            }

            populationFitness = population.CalculateFitnessByWanted(inputs, wantedOutputs);
            orderedPopulation = OrderByFitness(populationFitness);
            orderedByComplex = OrderByComplex();
        }

        auto simplifiedBestNN = population.generation.neuralNetwork.SimplifyRemoveDisable(*orderedPopulation[0]);
        auto compressedLeftBestNN = population.generation.neuralNetwork.SimplifyRemoveUselessConnectionLeft(simplifiedBestNN);
        auto compressedRightBestNN = population.generation.neuralNetwork.SimplifyRemoveUselessConnectionRight(compressedLeftBestNN);

        if (Opts.IterationCheckPoint > 0) {
            population.generation.neuralNetwork.ExportInnovations(Opts.CheckPointPath);
            population.generation.neuralNetwork.ExportNN(simplifiedBestNN, Opts.CheckPointPath); // 导出导入的格式定为没有已禁用连接，断点不需要简化孤立连接
        }

        population.generation.neuralNetwork.ExportNN(compressedRightBestNN, "./champion");
        population.generation.neuralNetwork.ExportNNToDot(compressedRightBestNN, "./champion");

        if (Opts.Enable3dNN) {
            std::cout << "需保持主线程不退出,防止3d显示bug\n";
        }

        return BestOne{.Gen = rounds, .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                //                        .NN = *orderedPopulation[0],
                .Fit = populationFitness[orderedPopulation[0]],};
    }

    BestOne SimpleNeat::TrainByInteractive(const std::function<std::map<NetworkGenome *, float>()> &interactiveFunc, const std::function<bool()> &isBreakFunc) {
        auto populationFitness = interactiveFunc();
        auto orderedPopulation = OrderByFitness(populationFitness);
        auto orderedByComplex = OrderByComplex();

        uint rounds = 1;
        float lastFitness = 0.f;

        for (; rounds <= Opts.IterationTimes || Opts.IterationTimes <= 0; ++rounds) {
            if (populationFitness[orderedPopulation[0]] > lastFitness || (Opts.IterationCheckPoint > 0 && rounds % Opts.IterationCheckPoint == 0)) {
                lastFitness = populationFitness[orderedPopulation[0]];
                std::cout << "gen: " << rounds << " ptr: " << orderedPopulation[0] << " age: " << orderedPopulation[0]->Age << " neurons: " << orderedPopulation[0]->Neurons.size() << " connections: "
                          << orderedPopulation[0]->Connections.size() << " fitness: " << populationFitness[orderedPopulation[0]] << "\n";
            }

            for (auto nn: orderedPopulation) {
                ++nn->Age;
            }

            if (Opts.FitnessThreshold > 0 && populationFitness[orderedPopulation[0]] >= Opts.FitnessThreshold || isBreakFunc()) {
                auto simplifiedBestNN = population.generation.neuralNetwork.SimplifyRemoveDisable(*orderedPopulation[0]);
                auto compressedLeftBestNN = population.generation.neuralNetwork.SimplifyRemoveUselessConnectionLeft(simplifiedBestNN);
                auto compressedRightBestNN = population.generation.neuralNetwork.SimplifyRemoveUselessConnectionRight(compressedLeftBestNN);

                if (Opts.IterationCheckPoint > 0) {
                    population.generation.neuralNetwork.ExportInnovations(Opts.CheckPointPath);
                    population.generation.neuralNetwork.ExportNN(simplifiedBestNN, Opts.CheckPointPath); // 导出导入的格式定为没有已禁用连接，断点不需要简化孤立连接
                }

                population.generation.neuralNetwork.ExportNN(compressedRightBestNN, "./champion");
                population.generation.neuralNetwork.ExportNNToDot(compressedRightBestNN, "./champion");

                if (Opts.Enable3dNN) {
                    tPool.push_task(Update3dNN, compressedRightBestNN, false);
                    std::cout << "需保持主线程不退出,防止3d显示bug\n";
                }

                return BestOne{.Gen = rounds, .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                        //                        .NN = *orderedPopulation[0],
                        .Fit = populationFitness[orderedPopulation[0]],};
            }

            if (Opts.IterationCheckPoint > 0 && rounds % Opts.IterationCheckPoint == 0) {
                auto simplifiedBestNN = population.generation.neuralNetwork.SimplifyRemoveDisable(*orderedPopulation[0]);

                if (Opts.IterationCheckPoint > 0) {
                    population.generation.neuralNetwork.ExportInnovations(Opts.CheckPointPath);
                    population.generation.neuralNetwork.ExportNN(simplifiedBestNN, Opts.CheckPointPath); // 导出导入的格式定为没有已禁用连接，断点不需要简化孤立连接
                }
            }

            std::vector<NetworkGenome> tmpPopulation(Opts.PopulationSize);
            std::vector<std::future<void>> thisFuture;// 如果用这个线程池的push_task函数，后面需要wait_for_tasks()，会卡死

            uint indexOutside = 0;
            for (auto &n: tmpPopulation) {
                thisFuture.push_back(tPool.submit([&](uint index, NetworkGenome *nn) {
                    if (index < Opts.ChampionToNewSize) {
                        *nn = *orderedPopulation[index % Opts.ChampionKeepSize];  // 选取ChampionKeepSize个个体填满前ChampionToNewSize个
                        if (index >= Opts.ChampionKeepSize && index < Opts.ChampionKeepSize * 2) {
                            population.generation.MutateNetworkGenome(*nn);  // 冠军一份副本进行变异
                        }
                        if (index >= Opts.ChampionKeepSize * 2) {
                            long choose = index % Opts.ChampionKeepSize;
                            auto nn0 = orderedPopulation[choose];
                            auto nn1 = orderedPopulation[choose + 1];
                            *nn = population.generation.GetChildByCrossing(nn0, nn1);
                        }
                    } else if (index < Opts.PopulationSize - Opts.NewSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        long chooseAnother = random() % Opts.PopulationSize;
                        auto nn0 = orderedPopulation[random() % Opts.ChampionKeepSize];
                        auto nn1 = orderedPopulation[chooseAnother];
//                        auto nn1 = orderedPopulation[Opts.ChampionKeepSize + random() % (Opts.PopulationSize - Opts.ChampionKeepSize)];
                        *nn = population.generation.GetChildByCrossing(nn0, nn1);
                        if (random() % 2 == 0 || nn0 == nn1 || chooseAnother < Opts.ChampionKeepSize) {
                            population.generation.MutateNetworkGenome(*nn);  // 繁殖以后进行变异
                        }
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        if (Opts.usingFCNN) {
                            *nn = population.generation.neuralNetwork.NewFCNN();
                        } else {
                            *nn = population.generation.neuralNetwork.NewNN();
                        }
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize) {
                        *nn = *orderedByComplex[index % Opts.KeepComplexSize];
                        population.generation.EnableAllConnections(*nn);
                    } else {
                        *nn = *orderedPopulation[index];
                        population.generation.MutateNetworkGenome(*nn);
                    }
                }, indexOutside, &n));
                ++indexOutside;
            }

            for (auto &f: thisFuture) {
                f.wait();
            }

            if (Opts.Enable3dNN) {
                tPool.push_task(Update3dNN, *orderedPopulation[0], false);
            }

            population.NeuralNetworks = tmpPopulation;

            populationFitness.clear();
            orderedPopulation.clear();
            orderedByComplex.clear();
            populationFitness = interactiveFunc();
            orderedPopulation = OrderByFitness(populationFitness);
            orderedByComplex = OrderByComplex();
        }

        auto simplifiedBestNN = population.generation.neuralNetwork.SimplifyRemoveDisable(*orderedPopulation[0]);
        auto compressedLeftBestNN = population.generation.neuralNetwork.SimplifyRemoveUselessConnectionLeft(simplifiedBestNN);
        auto compressedRightBestNN = population.generation.neuralNetwork.SimplifyRemoveUselessConnectionRight(compressedLeftBestNN);

        if (Opts.IterationCheckPoint > 0) {
            population.generation.neuralNetwork.ExportInnovations(Opts.CheckPointPath);
            population.generation.neuralNetwork.ExportNN(simplifiedBestNN, Opts.CheckPointPath); // 导出导入的格式定为没有已禁用连接，断点不需要简化孤立连接
        }

        population.generation.neuralNetwork.ExportNN(compressedRightBestNN, "./champion");
        population.generation.neuralNetwork.ExportNNToDot(compressedRightBestNN, "./champion");

        if (Opts.Enable3dNN) {
            std::cout << "需保持主线程不退出,防止3d显示bug\n";
        }

        return BestOne{.Gen = rounds, .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                //                        .NN = *orderedPopulation[0],
                .Fit = populationFitness[orderedPopulation[0]],};
    }
}

#endif //MYNEAT_SIMPLENEAT_HPP

// TODO channel