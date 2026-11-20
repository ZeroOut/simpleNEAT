#pragma once

#ifndef MYNEAT_GENERATION_HPP
#define MYNEAT_GENERATION_HPP

#include "NeuralNetwork.hpp"
#include <algorithm>  // std::sort
#include <set>  // [OPT] GetChildByCrossing 用 set/map 做连接匹配去重

namespace znn {
    class Generation {
    public:
        NeuralNetwork neuralNetwork;

        void MutateWeightDirect(Connection &c);

        void MutateWeightNear(Connection &c);

        void MutateBiasDirect(Neuron &o);

        void MutateBiasNear(Neuron &o);

        void MutateAddNeuron(NetworkGenome &nn);

        void MutateAddConnection(NetworkGenome &nn);

        void MutateEnableConnection(NetworkGenome &nn);

        void EnableAllConnections(NetworkGenome &nn);

        void MutateNetworkGenome(NetworkGenome &nn);

        NetworkGenome GetChildByCrossing(NetworkGenome *nn0, NetworkGenome *nn1);
    };

    // [FIX] 旧自由函数兼容层（test_mutate 等只含 Generation.hpp 的测试）
    inline Generation defaultGeneration;

    inline void MutateNetworkGenome(NetworkGenome &nn) {
        defaultGeneration.MutateNetworkGenome(nn);
    }

    void Generation::MutateWeightDirect(Connection &c) {
        c.Weight = float(random() % long(Opts.WeightRange * 2000000) - long(Opts.WeightRange * 1000000)) / 1000000.f;
    }

    void Generation::MutateWeightNear(Connection &c) {
        // [FIX] 原式 Opts.MutateWeightNearRange * 2000000 是 int 乘法，范围设置较大时会整型溢出；统一转 long 再乘
        float tmpWeight = c.Weight + float(random() % (long(Opts.MutateWeightNearRange) * 2000000L) - long(Opts.MutateWeightNearRange) * 1000000L) / 1000000.f;
        if (tmpWeight > Opts.WeightRange) {
            c.Weight = Opts.WeightRange;
            return;
        }
        if (tmpWeight < -Opts.WeightRange) {
            c.Weight = -Opts.WeightRange;
            return;
        }
        c.Weight = tmpWeight;
    }

    void Generation::MutateBiasDirect(Neuron &o) {
        o.Bias = float(random() % long(Opts.BiasRange * 2000000) - long(Opts.BiasRange * 1000000)) / 1000000;
    }

    void Generation::MutateBiasNear(Neuron &o) {
        float tmpBias = o.Bias + float(random() % long(Opts.MutateBiasNearRange * 2000000) - long(Opts.MutateBiasNearRange * 1000000)) / 1000000.f;
        if (tmpBias > Opts.BiasRange) {
            o.Bias = Opts.BiasRange;
            return;
        }
        if (tmpBias < -Opts.BiasRange) {
            o.Bias = -Opts.BiasRange;
            return;
        }
        o.Bias = tmpBias;
    }

    void Generation::MutateAddNeuron(NetworkGenome &nn) {
        ulong nid0, nid1;

        // [FIX] 原代码在 Connections / Neurons 为空时 random() % 0 未定义（除零崩溃）；
        // 空容器时直接返回，跳过本次变异
        if (nn.Connections.empty() || nn.Neurons.empty()) {
            return;
        }

        if (random() % 2 == 0) {
            Connection &choosingConnection = nn.Connections[random() % nn.Connections.size()];  // 选出一条连接

            //        if (!choosingConnection.Enable) {
            //            return;  // 如果连接被禁用就。。。算了不用这个判断
            //        }

            nid0 = choosingConnection.ConnectedNeuronId[0];  // 获取连接左边的神经元的id
            nid1 = choosingConnection.ConnectedNeuronId[1];  // 获取连接右边的神经元的id
        } else {
            Neuron &choosingNeuron0 = nn.Neurons[random() % nn.Neurons.size()];  // 选出第一个神经元
            Neuron &choosingNeuron1 = nn.Neurons[random() % nn.Neurons.size()];  // 选出第二个神经元

            if (choosingNeuron0.Layer == choosingNeuron1.Layer) {  // 同一layer包含 choosingNeuron0.Id == choosingNeuron1.Id
                return;
            }

            nid0 = choosingNeuron0.Id;
            nid1 = choosingNeuron1.Id;

            if (choosingNeuron0.Layer > choosingNeuron1.Layer) { // 保持从左到右编号
                nid0 = choosingNeuron1.Id;
                nid1 = choosingNeuron0.Id;
            }
        }

        std::unordered_map<ulong, Neuron *> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址
        for (auto &o: nn.Neurons) {
            tmpNeuronMap[o.Id] = &o;
        }

        mtx.lock();
        ulong newNid = neuralNetwork.HiddenNeuronInnovations.size() + Opts.InputSize + Opts.OutputSize + neuralNetwork.FCHidenNeuronSize;  // 新的神经元id为全部藏神经元数量+输入神经元数量+输出神经元数量+全连接网络隐藏神经元数量(如有)
        if (!neuralNetwork.HiddenNeuronInnovations.contains({nid0, nid1})) {  // 从全部隐藏神经元innovMap里面查看是否存在相同位置的神经元
//            neuralNetwork.HiddenNeuronInnovations.insert({{nid0, nid1}, newNid});
            neuralNetwork.HiddenNeuronInnovations[{nid0, nid1}] = newNid;  // 如果不存在则新增记录插入连接左右两个神经元id对应的隐藏层神经元id
        } else {
            //            std::vector<ulong> tmpRightNeuronIds;
            //            for (auto &n: nn.Connections) {
            //                if (n.ConnectedNeuronId[0] == nid0) {
            //                    tmpRightNeuronIds.push_back(n.ConnectedNeuronId[1]);
            //                }
            //            }
            //
            //            for (auto &n: nn.Connections) {
            //                if (n.ConnectedNeuronId[1] == nid1) {
            //                    for (auto &tnn: tmpRightNeuronIds) {
            //                        if (tnn == n.ConnectedNeuronId[0]) {
            //                            if (float(random() % 1000) / 1000.f < Opts.MutateAddNeuronRate) {
            //                                break;
            //                            } else {
            //                                return;  // 如果已经存在相同的连接左右神经元id的神经元，则不继续执行添加此神经元
            //                            }
            //                        }
            //                    }
            //                }
            //            }

//            if (random() % 100 < 1) {
////            if (random() % 100 > 0) {
//                //                    mtx.unlock();
//                //                    return;
//                //                } else {
//                neuralNetwork.HiddenNeuronInnovations.insert({{nid0, nid1}, newNid});
//                //                }
//            } else {
//                auto HI_Range = neuralNetwork.HiddenNeuronInnovations.equal_range({nid0, nid1});
//                std::mt19937 gen(random());
//                std::uniform_int_distribution<> dis(0, std::distance(HI_Range.first, HI_Range.second) - 1);
//                auto randomIt = std::next(HI_Range.first, dis(gen));
//                newNid = randomIt->second;  // 如果存在则使用已有隐藏层神经元id
            newNid = neuralNetwork.HiddenNeuronInnovations[{nid0, nid1}];
//            }
        }

        mtx.unlock();

        // [FIX] 本 genome 已有该 innovation 的神经元（连接已分裂过）时直接跳过：
        // 原实现会对已分裂的连接再次加神经元——innovation 表返回同一个 Id →
        // 出现重复 Id 神经元 + 重复连接（CPU 语义变求和翻倍、GPU supernet 槽位错乱）
        if (tmpNeuronMap.find(newNid) != tmpNeuronMap.end()) {
            return;
        }

        for (auto &c: nn.Connections) {
            if (c.ConnectedNeuronId[0] == nid0 && c.ConnectedNeuronId[1] == nid1) {
                c.Enable = false;
                break;
            }
        }

        // [FIX] 连接若引用了不存在的神经元（如手改过的 .nn 文件），tmpNeuronMap[nid] 用 operator[] 会插入
        // nullptr 再解引用 ->Layer 崩溃；改用 find 检查，缺失时报错跳过本次变异
        if (tmpNeuronMap.find(nid0) == tmpNeuronMap.end() || tmpNeuronMap.find(nid1) == tmpNeuronMap.end()) {
            std::cerr << "MutateAddNeuron: connection references missing neuron [" << nid0 << "," << nid1 << "]\n";
            return;
        }

        Neuron newNeuron = {.Id = newNid, .Bias = float(random() % long(Opts.BiasRange * 2000000) - long(Opts.BiasRange * 1000000)) / 1000000.f, .Layer =
        (tmpNeuronMap[nid0]->Layer + tmpNeuronMap[nid1]->Layer) / 2.,};
        nn.Neurons.push_back(newNeuron);

        Connection newConn0 = {  // 添加左侧连接
                .ConnectedNeuronId = {nid0, newNid,}, .Weight = float(random() % long(Opts.WeightRange * 2000000) - long(Opts.WeightRange * 1000000)) / 1000000.f, .Enable = true,};
        nn.Connections.push_back(newConn0);

        Connection newConn1 = {  // 添加右侧连接
                .ConnectedNeuronId = {newNid, nid1,}, .Weight = float(random() % long(Opts.WeightRange * 2000000) - long(Opts.WeightRange * 1000000)) / 1000000.f, .Enable = true,};
        nn.Connections.push_back(newConn1);

        //        mtx.lock();
        //        ConnectionInnovations[{nid0, newNid}] = ConnectionInnovations.size(); // 记录老的左侧神经元id和新神经元id对应的新连接innov
        //        ConnectionInnovations[{newNid, nid1}] = ConnectionInnovations.size(); // 记录老的新神经元id和右侧神经元id对应的新连接innov
        //        mtx.unlock();
    }

    void Generation::MutateAddConnection(NetworkGenome &nn) {
        // [FIX] Neurons 为空时 random() % 0 除零崩溃；空容器时直接返回
        if (nn.Neurons.empty()) {
            return;
        }

        Neuron &choosingNeuron0 = nn.Neurons[random() % nn.Neurons.size()];  // 选出第一个神经元
        Neuron &choosingNeuron1 = nn.Neurons[random() % nn.Neurons.size()];  // 选出第二个神经元

        if (choosingNeuron0.Layer == choosingNeuron1.Layer) {  // 同一layer包含 choosingNeuron0.Id == choosingNeuron1.Id
            return;
        }

        ulong nid0 = choosingNeuron0.Id;
        ulong nid1 = choosingNeuron1.Id;

        if (choosingNeuron0.Layer > choosingNeuron1.Layer) { // 保持从左到右编号
            nid0 = choosingNeuron1.Id;
            nid1 = choosingNeuron0.Id;
        }

        for (auto &c: nn.Connections) {
            if (c.ConnectedNeuronId[0] == nid0 && c.ConnectedNeuronId[1] == nid1) {
                c.Enable = true;
                return; // 已存在这个链接
            }
        }

        Connection newConn = {  // 添加连接
                .ConnectedNeuronId = {nid0, nid1,}, .Weight = float(random() % long(Opts.WeightRange * 2000000) - long(Opts.WeightRange * 1000000)) / 1000000.f, .Enable = true,};
        nn.Connections.push_back(newConn);

        //        mtx.lock();
        //        if (ConnectionInnovations.find({nid0, nid1}) == ConnectionInnovations.end()) {  // 查询所有连接的innov是否存在
        //            ConnectionInnovations[{nid0, nid1}] = ConnectionInnovations.size();  // 如果不存在则新增连接的innov
        //        }
        //        mtx.unlock();
    }

    void Generation::MutateEnableConnection(NetworkGenome &nn) {
        // [FIX] Connections 为空时 random() % 0 除零崩溃；空容器时直接返回
        if (nn.Connections.empty()) {
            return;
        }

        Connection &choosingConnection = nn.Connections[random() % nn.Connections.size()];  // 选出一条连接
        if (choosingConnection.Enable) {
            choosingConnection.Enable = false;
            return;
        }
        choosingConnection.Enable = true;
    }

    void Generation::EnableAllConnections(NetworkGenome &nn) {
        for (auto &c: nn.Connections) {
            if (!c.Enable) {
                c.Enable = true;
            }
        }
    }

    void Generation::MutateNetworkGenome(NetworkGenome &nn) {
        for (auto &c: nn.Connections) {
            if (float(random() % 1000) / 1000.f < Opts.MutateWeightRate) {
                if (float(random() % 1000) / 1000.f < Opts.MutateWeightDirectOrNear) {
                    MutateWeightDirect(c);
                } else {
                    MutateWeightNear(c);
                }
                //                nn.Age = 0;
            }
        }

        for (auto &n: nn.Neurons) {
            if (float(random() % 1000) / 1000.f < Opts.MutateBiasRate && n.Layer > 0.) {
                if (float(random() % 1000) / 1000.f < Opts.MutateBiasDirectOrNear) {
                    MutateBiasDirect(n);
                } else {
                    MutateBiasNear(n);
                }
                //                nn.Age = 0;
            }
        }

        if (float(random() % 1000) / 1000.f < Opts.MutateAddNeuronRate) {
            MutateAddNeuron(nn);
            //            nn.Age = 0;
        }

        if (float(random() % 1000) / 1000.f < Opts.MutateAddConnectionRate) {
            MutateAddConnection(nn);
            //            nn.Age = 0;
        }

        if (float(random() % 1000) / 1000.f < Opts.MutateEnableConnectionRate) {
            MutateEnableConnection(nn);
            //            nn.Age = 0;
        }
    }

    NetworkGenome Generation::GetChildByCrossing(NetworkGenome *nn0, NetworkGenome *nn1) {
        if (float(random() % 1000) / 1000.f > Opts.CrossoverRate || nn0 == nn1) {
            return *nn1; // nn0 是冠军中的个体， nn1 是剩余的，冠军已经保留了原始基因，所以按照概率保留非冠军基因
        }

        std::vector<Connection> newConnections;// 记录全部涉及的连接

        // [OPT] 原实现三层嵌套循环匹配连接：nn0 每条连接线性扫 nn1（O(C0×C1)），再线性扫
        // newConnections 去重（O(C0×C2)）。改为 map 按"两端神经元id"键 O(ClogC) 查找，
        // 选择语义与原版完全一致（共同连接随机选一条、独有连接直接保留）
        std::map<std::array<ulong, 2>, const Connection *> conn1Map;  // nn1 的连接按端点对索引
        for (auto &c1: nn1->Connections) {
            conn1Map[c1.ConnectedNeuronId] = &c1;
        }
        std::set<std::array<ulong, 2>> chosenKeys;  // 已选入 newConnections 的端点对，用于第二阶段去重

        for (auto c0: nn0->Connections) {  // 遍历第一个神经元的所有连接
            auto it1 = conn1Map.find(c0.ConnectedNeuronId);
            if (it1 != conn1Map.end()) {  // 两亲本共有此连接：随机选一条
                if (random() % 2 == 0) {
                    newConnections.push_back(c0);
                } else {
                    newConnections.push_back(*it1->second);
                }
            } else {
                newConnections.push_back(c0);// 仅 nn0 有的连接，直接保留
            }
            chosenKeys.insert(c0.ConnectedNeuronId);
        }

        for (auto c1: nn1->Connections) {// 遍历第二个神经元的所有连接
            if (chosenKeys.find(c1.ConnectedNeuronId) == chosenKeys.end()) {
                newConnections.push_back(c1);// 仅 nn1 有的连接，保留
                chosenKeys.insert(c1.ConnectedNeuronId);
            }
        } // 选出全部连接

        std::unordered_map<ulong, uint> remainingIds;  // 记录所有涉及的神经元id
        std::unordered_map<ulong, Neuron *> tmpNeuron0Map;

        for (auto &n: nn0->Neurons) {
            tmpNeuron0Map[n.Id] = &n;
            remainingIds[n.Id] = 0;
        }

        std::unordered_map<ulong, Neuron *> tmpNeuron1Map;
        for (auto &n: nn1->Neurons) {
            tmpNeuron1Map[n.Id] = &n;
            remainingIds[n.Id] = 0;
        }

        std::vector<Neuron> newNeurons;

        for (auto &i: remainingIds) {
            bool isNeuronIn0 = (tmpNeuron0Map.find(i.first) != tmpNeuron0Map.end());
            //            bool isNeuronIn0 = (tmpNeuron0Map.contains(i.first));
            bool isNeuronIn1 = (tmpNeuron1Map.find(i.first) != tmpNeuron1Map.end());
            //            bool isNeuronIn1 = (tmpNeuron1Map.contains(i.first));

            if (isNeuronIn0 && isNeuronIn1) {
                if (random() % 2 == 0) {
                    newNeurons.push_back(*tmpNeuron0Map[i.first]);
                } else {
                    newNeurons.push_back(*tmpNeuron1Map[i.first]);
                }
                continue;
            }

            if (!isNeuronIn0) {
                newNeurons.push_back(*tmpNeuron1Map[i.first]);
                continue;
            }

            newNeurons.push_back(*tmpNeuron0Map[i.first]);
        }

        // [FIX] 按 Id 排序：remainingIds 是 unordered_map，迭代序随机 → 子代 Neurons 向量顺序
        // 每次都不同（配合 FeedForwardPredict 的向量序输入分配会导致输入映射漂移；导出文件也不稳定）
        std::sort(newNeurons.begin(), newNeurons.end(), [](const Neuron &a, const Neuron &b) { return a.Id < b.Id; });

        return NetworkGenome{.Neurons = newNeurons, .Connections = newConnections,};
    }

}

#endif //MYNEAT_GENERATION_HPP
