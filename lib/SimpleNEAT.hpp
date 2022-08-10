#pragma once

#ifndef MYNEAT_MYNEAT_HPP
#define MYNEAT_MYNEAT_HPP

#include "Population.hpp"

namespace znn {
    void CheckOptions() {
        if (Opts.ChampionToNewSize + Opts.KeepWorstSize + Opts.NewSize + Opts.KeepComplexSize > Opts.PopulationSize) {
            std::cerr << "Opts.ChampionToNewSize + Opts.KeepLastSize + Opts.NewSize + Opts.KeepComplexSize > Opts.PopulationSize" << std::endl;
            exit(0);
        }

        if (Opts.ChampionKeepSize > Opts.ChampionToNewSize) {
            std::cerr << "Opts.ChampionKeepSize > Opts.ChampionToNewSize" << std::endl;
            exit(0);
        }

        if (Opts.FitnessThreshold <= 0 && Opts.IterationTimes <= 0) {
            std::cerr << "Opts.FitnessThreshold <= 0, Opts.IterationTimes <= 0, cannot <= 0 both" << std::endl;
            exit(0);
        }
    }

    void StartNew() {
        CheckOptions();
        CreatePopulation();
    }

    void StartWithCheckPoint() {
        CheckOptions();
        CreatePopulationByGiving();
    }

    void Start() {
        if (access((Opts.CheckPointPath + ".innov").c_str(), F_OK) != -1 && access((Opts.CheckPointPath + ".nn").c_str(), F_OK) != -1) {
            StartWithCheckPoint();
        } else {
            std::clog << "Check point files are not exist, start new.\n";
            StartNew();
        }
    }

    bool cmpf(std::pair<NetworkGenome *, float> &a, std::pair<NetworkGenome *, float> &b) {
        return a.second > b.second;  // 从大到小排列
    }

    std::vector<NetworkGenome *> OrderByFitness(std::map<NetworkGenome *, float> &M) {  // Comparator function to sort pairs according to second value
        std::vector<NetworkGenome *> result;
        std::vector<std::pair<NetworkGenome *, float> > A;  // Declare vector of pairs
        for (auto &it : M) {  // Copy key-value pair from Map to vector of pairs
            A.push_back(it);
        }
        std::sort(A.begin(), A.end(), cmpf);  // Sort using comparator function
        for (auto &it : A) {
            result.push_back(it.first);
        }
        return result;
    }

    bool cmpc(std::pair<NetworkGenome *, uint> &a, std::pair<NetworkGenome *, uint> &b) {
        return a.second > b.second;  // 从大到小排列
    }

    std::vector<NetworkGenome *> OrderByComplex() {  // Comparator function to sort pairs according to second value
        std::vector<NetworkGenome *> result;
        std::vector<std::pair<NetworkGenome *, uint> > A;  // Declare vector of pairs
        for (auto &it : Population) {  // Copy key-value pair from Map to vector of pairs
            A.push_back(std::pair{&it, it.Neurons.size()});
        }
        std::sort(A.begin(), A.end(), cmpc);  // Sort using comparator function
        for (auto &it : A) {
            result.push_back(it.first);
        }
        return result;
    }

    struct BestOne {
        uint Gen{};
        NetworkGenome NN;
        float Fit{};
    };

    BestOne TrainByWanted(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &wantedOutputs) {
        auto populationFitness = CalculateFitnessByWanted(inputs, wantedOutputs);
        auto orderedPopulation = OrderByFitness(populationFitness);
        auto orderedByComplex = OrderByComplex();

        uint rounds = 1;
        float lastFitness = 0.f;

        for (; rounds <= Opts.IterationTimes || Opts.IterationTimes <= 0; ++rounds) {
            srandom((unsigned) clock());

            if (populationFitness[orderedPopulation[0]] > lastFitness) {
                lastFitness = populationFitness[orderedPopulation[0]];
                std::cout << "gen: " << rounds << " " << orderedPopulation[0] << " " << orderedPopulation[0]->Neurons.size() << " " << orderedPopulation[0]->Connections.size() << " fitness: "
                          << populationFitness[orderedPopulation[0]] << " " << std::endl;
            }

            if (Opts.FitnessThreshold > 0 && populationFitness[orderedPopulation[0]] >= Opts.FitnessThreshold) {
                auto simplifiedBestNN = SimplifyRemoveDisable(*orderedPopulation[0]);
                auto compressedLeftBestNN = SimplifyRemoveUselessConnectionLeft(simplifiedBestNN);
                auto compressedRightBestNN = SimplifyRemoveUselessConnectionRight(compressedLeftBestNN);

                if (Opts.IterationCheckPoint > 0) {
                    ExportInnovations(Opts.CheckPointPath);
                    ExportNetwork(simplifiedBestNN, Opts.CheckPointPath); // 导出导入的格式定为没有已禁用连接，断点不需要简化孤立连接
                }

                ExportNetwork(compressedRightBestNN, "./champion");
                ExportNetworkToDot(compressedRightBestNN, "./champion");

                return BestOne{
                        .Gen = rounds,
                        .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
//                        .NN = *orderedPopulation[0],
                        .Fit = populationFitness[orderedPopulation[0]],
                };
            }

            if (Opts.IterationCheckPoint > 0 && rounds % Opts.IterationCheckPoint == 0) {
                auto simplifiedBestNN = SimplifyRemoveDisable(*orderedPopulation[0]);

                if (Opts.IterationCheckPoint > 0) {
                    ExportInnovations(Opts.CheckPointPath);
                    ExportNetwork(simplifiedBestNN, Opts.CheckPointPath); // 导出导入的格式定为没有已禁用连接，断点不需要简化孤立连接
                }
            }

            std::vector<NetworkGenome> tmpPopulation(Opts.PopulationSize);
            std::vector<std::future<void>> thisFuture;  // 如果用这个线程池的push_task函数，后面需要wait_for_tasks()，会卡死

            uint indexOutside = 0;
            for (auto &nn : tmpPopulation) {
                thisFuture.push_back(tPool.submit([&]() {
                    mtx.lock();
                    uint index = indexOutside;
                    ++indexOutside;
                    mtx.unlock();

                    if (index < Opts.ChampionToNewSize) {
                        nn = *orderedPopulation[index % Opts.ChampionKeepSize];  // 选取ChampionKeepSize个个体填满前ChampionToNewSize个
                        if (index >= Opts.ChampionKeepSize) {
                            MutateNetworkGenome(nn);  // 除开原始冠军，他们的克隆体进行变异
                        }
                    } else if (index < Opts.PopulationSize - Opts.NewSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        auto nn0 = orderedPopulation[random() % Opts.ChampionKeepSize];
                        auto nn1 = orderedPopulation[Opts.ChampionKeepSize + random() % (Opts.PopulationSize - Opts.ChampionKeepSize)];
                        nn = GetChildByCrossing(nn0, nn1);
                        if ((index % 2 == 0 || nn0 == nn1) && nn0->Neurons.size() < orderedByComplex[0]->Neurons.size() && nn1->Neurons.size() < orderedByComplex[0]->Neurons.size()) {
                            MutateNetworkGenome(nn);  // 繁殖以后进行变异
                        }
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                        nn = NewNN();
                    } else if (index < Opts.PopulationSize - Opts.KeepWorstSize) {
                        nn = *orderedByComplex[index % Opts.KeepComplexSize];
                        EnableAllConnections(nn);
                    } else {
                        nn = *orderedPopulation[index];
                        MutateNetworkGenome(nn);
                    }
                }));
            }

            for (auto &f : thisFuture) {
                f.wait();
            }

            Population = tmpPopulation;

            populationFitness.clear();
            orderedPopulation.clear();
            orderedByComplex.clear();
            populationFitness = CalculateFitnessByWanted(inputs, wantedOutputs);
            orderedPopulation = OrderByFitness(populationFitness);
            orderedByComplex = OrderByComplex();
        }

        auto simplifiedBestNN = SimplifyRemoveDisable(*orderedPopulation[0]);
        auto compressedLeftBestNN = SimplifyRemoveUselessConnectionLeft(simplifiedBestNN);
        auto compressedRightBestNN = SimplifyRemoveUselessConnectionRight(compressedLeftBestNN);

        if (Opts.IterationCheckPoint > 0) {
            ExportInnovations(Opts.CheckPointPath);
            ExportNetwork(simplifiedBestNN, Opts.CheckPointPath); // 导出导入的格式定为没有已禁用连接，断点不需要简化孤立连接
        }

        ExportNetwork(compressedRightBestNN, "./champion");
        ExportNetworkToDot(compressedRightBestNN, "./champion");

        return BestOne{
                .Gen = rounds,
                .NN = compressedRightBestNN, // 导出导入的格式定为没有已禁用连接
                //                        .NN = *orderedPopulation[0],
                .Fit = populationFitness[orderedPopulation[0]],
        };
    }
}

#endif //MYNEAT_MYNEAT_HPP

// TODO: channel