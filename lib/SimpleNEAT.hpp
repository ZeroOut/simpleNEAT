#pragma once

#ifndef MYNEAT_SIMPLENEAT_HPP
#define MYNEAT_SIMPLENEAT_HPP

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

        void StartNewFC(std::vector<int> hideLayers);

        void StartWithCheckPoint();

        void Start();

        znn::BestOne TrainByWanted(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &wantedOutputs);

        znn::BestOne TrainByWantedRandom(const std::vector<std::vector<float>> &rawInputs, const std::vector<std::vector<float>> &rawWantedOutputs, const uint chooseSize);

        std::vector<NetworkGenome *> OrderByFitness(std::map<NetworkGenome *, float> &M);

        std::vector<NetworkGenome *> OrderByComplex();

        BestOne TrainByInteractive(const std::function<std::map<NetworkGenome *, float>()> &interactiveFunc, const std::function<bool()> &isBreakFunc);
    };


    void CheckOptions() {
        if (Opts.ChampionToNewSize + Opts.KeepWorstSize + Opts.NewSize + Opts.KeepComplexSize > Opts.PopulationSize) {
            std::cerr << "Opts.ChampionToNewSize + Opts.KeepWorstSize + Opts.NewSize + Opts.KeepComplexSize > Opts.PopulationSize" << std::endl;
            exit(0);
        }

        if (Opts.ChampionKeepSize * 2 > Opts.ChampionToNewSize) {
            std::cerr << "Opts.ChampionKeepSize * 2 > Opts.ChampionToNewSize" << std::endl;
            exit(0);
        }

        if (Opts.FitnessThreshold <= 0 && Opts.IterationTimes <= 0) {
            std::clog << "Warning: Opts.FitnessThreshold <= 0, Opts.IterationTimes <= 0, will be infinity loop" << std::endl;
        }

        tPool.reset(Opts.ThreadCount);
        srandom((unsigned) clock());

        if (Opts.Enable3dNN) {
            tPool.push_task(Show3dNN);
        }
    }

    void SimpleNeat::StartNew() {
        CheckOptions();
        population.CreatePopulation();
    }

    void SimpleNeat::StartNewFC(std::vector<int> hideLayers) {
        CheckOptions();
        population.CreatePopulationFC(hideLayers);
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
        for (auto &it : M) {  // Copy key-value pair from Map to vector of pairs
            A.push_back(it);
        }
        std::sort(A.begin(), A.end(), cmpf);// Sort using comparator function
        for (auto &it : A) {
            result.push_back(it.first);
        }
        return result;
    }

    bool cmpc(std::pair<NetworkGenome *, uint> &a, std::pair<NetworkGenome *, uint> &b) {
        return a.second > b.second;// 从大到小排列
    }

    std::vector<NetworkGenome *> SimpleNeat::OrderByComplex() {  // Comparator function to sort pairs according to second value
        std::vector<NetworkGenome *> result;
        std::vector<std::pair<NetworkGenome *, uint> > A;// Declare vector of pairs
        for (auto &it : population.NeuralNetworks) {  // Copy key-value pair from Map to vector of pairs
            A.push_back(std::pair{&it, it.Connections.size()});
        }
        std::sort(A.begin(), A.end(), cmpc);// Sort using comparator function
        for (auto &it : A) {
            result.push_back(it.first);
        }
        return result;
    }

    BestOne SimpleNeat::TrainByWanted(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &wantedOutputs) {
        auto populationFitness = population.CalculateFitnessByWanted(inputs, wantedOutputs);
        auto orderedPopulation = OrderByFitness(populationFitness);
        auto orderedByComplex = OrderByComplex();

        uint rounds = 1;
        float lastFitness = 0.f;

        for (; rounds <= Opts.IterationTimes || Opts.IterationTimes <= 0; ++rounds) {
            //            srandom((unsigned) clock());

            if (populationFitness[orderedPopulation[0]] > lastFitness || (Opts.IterationCheckPoint > 0 && rounds % Opts.IterationCheckPoint == 0)) {
                lastFitness = populationFitness[orderedPopulation[0]];
                std::cout << "gen: " << rounds << " " << orderedPopulation[0] << " " << orderedPopulation[0]->Neurons.size() << " " << orderedPopulation[0]->Connections.size() << " fitness: "
                          << populationFitness[orderedPopulation[0]] << " " << std::endl;
            }

            if (Opts.FitnessThreshold > 0 && populationFitness[orderedPopulation[0]] >= Opts.FitnessThreshold) {
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
                    while (update3dLock) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    Update3dNN(compressedRightBestNN);
                    std::cout << "需保持主线程不退出,防止3d显示bug\n";
                }

                return BestOne{
                        .Gen = rounds,
                        .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                        //                        .NN = *orderedPopulation[0],
                        .Fit = populationFitness[orderedPopulation[0]],
                };
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
            for (auto &nn : tmpPopulation) {
                thisFuture.push_back(tPool.submit([&]() {
                    mtx.lock();
                    uint index = indexOutside;
                    ++indexOutside;
                    mtx.unlock();

                    if (index < Opts.ChampionToNewSize) {
                        nn = *orderedPopulation[index % Opts.ChampionKeepSize];// 选取ChampionKeepSize个个体填满前ChampionToNewSize个
                        if (index >= Opts.ChampionKeepSize && index < Opts.ChampionKeepSize * 2) {
                            for (uint i = 0; i < inputs.size(); ++i) {  // 保留的冠军一份副本全部进行反向传播更新weight和bias
                                population.generation.BackPropagation(&nn, inputs[i], wantedOutputs[i]);
                            }
                        }
                        if (index >= Opts.ChampionKeepSize * 2) {
                            population.generation.MutateNetworkGenome(nn);// 除开原始冠军，他们的克隆体进行变异
                        }
                    } else if (index < Opts.PopulationSize - Opts.NewSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        auto nn0 = orderedPopulation[random() % Opts.ChampionKeepSize];
                        //                        auto nn1 = orderedPopulation[random() % Opts.ChampionKeepSize];
                        auto nn1 = orderedPopulation[Opts.ChampionKeepSize + random() % (Opts.PopulationSize - Opts.ChampionKeepSize)];
                        nn = population.generation.GetChildByCrossing(nn0, nn1);
                        if ((index % 2 == 0 || nn0 == nn1) && nn0->Neurons.size() < orderedByComplex[0]->Neurons.size() && nn1->Neurons.size() < orderedByComplex[0]->Neurons.size()) {
                            population.generation.MutateNetworkGenome(nn);// 繁殖以后进行变异
                        }
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        nn = population.generation.neuralNetwork.NewNN();
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize) {
                        nn = *orderedByComplex[index % Opts.KeepComplexSize];
                        population.generation.EnableAllConnections(nn);
                    } else {
                        nn = *orderedPopulation[index];
                        population.generation.MutateNetworkGenome(nn);
                    }
                }));
            }

            for (auto &f : thisFuture) {
                f.wait();
            }

            if (Opts.Enable3dNN) {
                tPool.push_task([&]() {
                    Update3dNN(*orderedPopulation[0]);
                });
            }

            population.NeuralNetworks = tmpPopulation;

            populationFitness.clear();
            orderedPopulation.clear();
            orderedByComplex.clear();
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

        return BestOne{
                .Gen = rounds,
                .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                //                        .NN = *orderedPopulation[0],
                .Fit = populationFitness[orderedPopulation[0]],
        };
    }

    BestOne SimpleNeat::TrainByWantedRandom(const std::vector<std::vector<float>> &rawInputs, const std::vector<std::vector<float>> &rawWantedOutputs, const uint chooseSize) {
        if (rawInputs.size() != rawWantedOutputs.size()) {
            std::cerr << "rawInputs size: " << rawInputs.size() << " != rawWantedOutputs size: " << rawWantedOutputs.size() << "\n";
            exit(0);
        }

        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<float>> wantedOutputs;

        for (uint i = 0; i < chooseSize; ++i) {
            auto chooseIndex = random() % rawInputs.size();
            inputs.push_back(rawInputs[chooseIndex]);
            wantedOutputs.push_back(rawWantedOutputs[chooseIndex]);
        }

        auto populationFitness = population.CalculateFitnessByWanted(inputs, wantedOutputs);
        auto orderedPopulation = OrderByFitness(populationFitness);
        auto orderedByComplex = OrderByComplex();

        uint rounds = 1;
        float lastFitness = 0.f;

        for (; rounds <= Opts.IterationTimes || Opts.IterationTimes <= 0; ++rounds) {
            //            srandom((unsigned) clock());

            if (populationFitness[orderedPopulation[0]] > lastFitness || (Opts.IterationCheckPoint > 0 && rounds % Opts.IterationCheckPoint == 0)) {
                lastFitness = populationFitness[orderedPopulation[0]];
                std::cout << "gen: " << rounds << " " << orderedPopulation[0] << " " << orderedPopulation[0]->Neurons.size() << " " << orderedPopulation[0]->Connections.size() << " fitness: "
                          << populationFitness[orderedPopulation[0]] << " " << std::endl;
            }

            if (Opts.FitnessThreshold > 0 && populationFitness[orderedPopulation[0]] >= Opts.FitnessThreshold) {
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
                    while (update3dLock) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    Update3dNN(compressedRightBestNN);
                    std::cout << "需保持主线程不退出,防止3d显示bug\n";
                }

                return BestOne{
                        .Gen = rounds,
                        .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                        //                        .NN = *orderedPopulation[0],
                        .Fit = populationFitness[orderedPopulation[0]],
                };
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
            for (auto &nn : tmpPopulation) {
                thisFuture.push_back(tPool.submit([&]() {
                    mtx.lock();
                    uint index = indexOutside;
                    ++indexOutside;
                    mtx.unlock();

                    if (index < Opts.ChampionToNewSize) {
                        nn = *orderedPopulation[index % Opts.ChampionKeepSize];// 选取ChampionKeepSize个个体填满前ChampionToNewSize个
                        if (index >= Opts.ChampionKeepSize && index < Opts.ChampionKeepSize * 2) {
                            for (uint i = 0; i < inputs.size(); ++i) {  // 保留的冠军一份副本全部进行反向传播更新weight和bias
                                population.generation.BackPropagation(&nn, inputs[i], wantedOutputs[i]);
                            }
                        }
                        if (index >= Opts.ChampionKeepSize * 2) {
                            population.generation.MutateNetworkGenome(nn);// 除开原始冠军，他们的克隆体进行变异
                        }
                    } else if (index < Opts.PopulationSize - Opts.NewSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        auto nn0 = orderedPopulation[random() % Opts.ChampionKeepSize];
                        auto nn1 = orderedPopulation[Opts.ChampionKeepSize + random() % (Opts.PopulationSize - Opts.ChampionKeepSize)];
                        nn = population.generation.GetChildByCrossing(nn0, nn1);
                        if ((index % 2 == 0 || nn0 == nn1) && nn0->Neurons.size() < orderedByComplex[0]->Neurons.size() && nn1->Neurons.size() < orderedByComplex[0]->Neurons.size()) {
                            population.generation.MutateNetworkGenome(nn);// 繁殖以后进行变异
                        }
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        nn = population.generation.neuralNetwork.NewNN();
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize) {
                        nn = *orderedByComplex[index % Opts.KeepComplexSize];
                        population.generation.EnableAllConnections(nn);
                    } else {
                        nn = *orderedPopulation[index];
                        population.generation.MutateNetworkGenome(nn);
                    }
                }));
            }

            for (auto &f : thisFuture) {
                f.wait();
            }


            if (Opts.Enable3dNN) {
                tPool.push_task([&]() {
                    Update3dNN(*orderedPopulation[0]);
                });
            }

            population.NeuralNetworks = tmpPopulation;

            populationFitness.clear();
            orderedPopulation.clear();
            orderedByComplex.clear();

            inputs.clear();
            wantedOutputs.clear();

            for (uint i = 0; i < chooseSize; ++i) {
                auto chooseIndex = random() % rawInputs.size();
                inputs.push_back(rawInputs[chooseIndex]);
                wantedOutputs.push_back(rawWantedOutputs[chooseIndex]);
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

        return BestOne{
                .Gen = rounds,
                .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                //                        .NN = *orderedPopulation[0],
                .Fit = populationFitness[orderedPopulation[0]],
        };
    }

    BestOne SimpleNeat::TrainByInteractive(const std::function<std::map<NetworkGenome *, float>()> &interactiveFunc, const std::function<bool()> &isBreakFunc) {
        auto populationFitness = interactiveFunc();
        auto orderedPopulation = OrderByFitness(populationFitness);
        auto orderedByComplex = OrderByComplex();

        uint rounds = 1;
        float lastFitness = 0.f;

        for (; rounds <= Opts.IterationTimes || Opts.IterationTimes <= 0; ++rounds) {
            //            srandom((unsigned) clock());

            if (populationFitness[orderedPopulation[0]] > lastFitness || (Opts.IterationCheckPoint > 0 && rounds % Opts.IterationCheckPoint == 0)) {
                lastFitness = populationFitness[orderedPopulation[0]];
                std::cout << "gen: " << rounds << " " << orderedPopulation[0] << " " << orderedPopulation[0]->Neurons.size() << " " << orderedPopulation[0]->Connections.size() << " fitness: "
                          << populationFitness[orderedPopulation[0]] << " " << std::endl;
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
                    while (update3dLock) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    Update3dNN(compressedRightBestNN);
                    std::cout << "需保持主线程不退出,防止3d显示bug\n";
                }

                return BestOne{
                        .Gen = rounds,
                        .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                        //                        .NN = *orderedPopulation[0],
                        .Fit = populationFitness[orderedPopulation[0]],
                };
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
            for (auto &nn : tmpPopulation) {
                thisFuture.push_back(tPool.submit([&]() {
                    mtx.lock();
                    uint index = indexOutside;
                    ++indexOutside;
                    mtx.unlock();

                    if (index < Opts.ChampionToNewSize) {
                        nn = *orderedPopulation[index % Opts.ChampionKeepSize];// 选取ChampionKeepSize个个体填满前ChampionToNewSize个
                        if (index >= Opts.ChampionKeepSize && index < Opts.ChampionKeepSize * 2) {
                            auto nn0 = orderedPopulation[Opts.ChampionKeepSize % (index - (random() % (Opts.ChampionKeepSize - 1)))];  // 保留的冠军一份副本互相交配
                            nn = population.generation.GetChildByCrossing(nn0, &nn);
                        }
                        if (index >= Opts.ChampionKeepSize * 2) {
                            population.generation.MutateNetworkGenome(nn);// 除开原始冠军，他们的克隆体进行变异
                        }
                    } else if (index < Opts.PopulationSize - Opts.NewSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        auto nn0 = orderedPopulation[random() % Opts.ChampionKeepSize];
                        //                        auto nn1 = orderedPopulation[random() % Opts.ChampionKeepSize];
                        auto nn1 = orderedPopulation[Opts.ChampionKeepSize + random() % (Opts.PopulationSize - Opts.ChampionKeepSize)];
                        nn = population.generation.GetChildByCrossing(nn0, nn1);
                        if ((index % 2 == 0 || nn0 == nn1) && nn0->Neurons.size() < orderedByComplex[0]->Neurons.size() && nn1->Neurons.size() < orderedByComplex[0]->Neurons.size()) {
                            population.generation.MutateNetworkGenome(nn);// 繁殖以后进行变异
                        }
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        nn = population.generation.neuralNetwork.NewNN();
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize) {
                        nn = *orderedByComplex[index % Opts.KeepComplexSize];
                        population.generation.EnableAllConnections(nn);
                    } else {
                        nn = *orderedPopulation[index];
                        population.generation.MutateNetworkGenome(nn);
                    }
                }));
            }

            for (auto &f : thisFuture) {
                f.wait();
            }

            if (Opts.Enable3dNN) {
                tPool.push_task([&]() {
                    Update3dNN(*orderedPopulation[0]);
                });
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

        return BestOne{
                .Gen = rounds,
                .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                //                        .NN = *orderedPopulation[0],
                .Fit = populationFitness[orderedPopulation[0]],
        };
    }
}

#endif //MYNEAT_SIMPLENEAT_HPP

// TODO channel