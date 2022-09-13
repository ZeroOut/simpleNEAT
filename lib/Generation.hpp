#pragma once

#ifndef MYNEAT_GENERATION_HPP
#define MYNEAT_GENERATION_HPP

#include "NeuralNetwork.hpp"

namespace znn {
    class Generation {
    public:
        NeuralNetwork neuralNetwork;

        bool BackPropagation(NetworkGenome *nn, std::vector<float> inputs, std::vector<float> wants);

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

    bool Generation::BackPropagation(NetworkGenome *nn, std::vector<float> inputs, std::vector<float> wants) {  // 如果当前预测fitness大于预设，则判断为解决问题，返回true TODO: 权重和偏置范围该怎么限制?丢弃?
        std::map<uint, Neuron *> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址
        std::map<float, std::vector<Neuron *>> tmpLayerMap;  // 记录层对应神经元，同上因为记录的是神经元地址，需要的时候才能临时生成记录

        for (auto &n : nn->Neurons) {
            tmpNeuronMap[n.Id] = &n;
            tmpLayerMap[n.Layer].push_back(&n);
        }


        if (Opts.InputSize != inputs.size()) {
            std::cerr << "Input length " << inputs.size() << " diffrent with NN input nodes " << Opts.InputSize << std::endl;
            std::exit(0);
        }

        std::map<uint, float> tmpNodesOutput;
        std::map<uint, float> tmpNodesInput;

        std::function<void(uint)> calculateNeuron = [&](uint nid) {
            tmpNodesInput[nid] = 0.f;
            for (auto &connection : nn->Connections) {
                if (connection.ConnectedNeuronId[1] == nid && connection.Enable) {
                    tmpNodesInput[nid] += tmpNodesOutput[connection.ConnectedNeuronId[0]] * connection.Weight;
                }
            }
            tmpNodesInput[nid] += tmpNeuronMap[nid]->Bias;
            tmpNodesOutput[nid] = Opts.ActiveFunction(tmpNodesInput[nid]);
        };

        std::vector<float> outputs;

        for (auto &l : tmpLayerMap) {    // 神经元根据layer排序
            for (auto &n : l.second) {
                if (l.first == 0.f) {   // 初始化输入节点
                    tmpNodesOutput[n->Id] = inputs[n->Id];
                    continue;
                }

                calculateNeuron(n->Id);  // 计算隐藏神经元和输出神经元

                if (l.first == 1.f) {
                    outputs.push_back(tmpNodesOutput[n->Id]);
                }
            }
        }

        // 上面是正向计算全部节点输出,接下来开始反向传播
        // 先计算每个节点的误差,偏导数又称为误差项也称为灵敏度

        std::map<uint, float> tmpNodesOutputError;

        std::function<void(uint)> calculateNeuronError = [&](uint nid) {
            tmpNodesOutputError[nid] = 0.f;
            for (auto &connection : nn->Connections) {
                if (connection.ConnectedNeuronId[0] == nid && connection.Enable) {
                    tmpNodesOutputError[nid] += tmpNodesOutputError[connection.ConnectedNeuronId[1]] * connection.Weight;
                }
            }
            tmpNodesOutputError[nid] *= Opts.DerivativeFunction(tmpNodesOutput[nid]) * tmpNodesOutput[nid];
        };

        uint wantsCount = 0;
        for (std::map<float, std::vector<Neuron *>>::reverse_iterator ri = tmpLayerMap.rbegin(); ri != tmpLayerMap.rend(); ++ri) {
            for (auto &n : ri->second) {
                if (ri->first == 1.f) {   // 计算输出神经元点误差
                    tmpNodesOutputError[n->Id] = Opts.DerivativeFunction(tmpNodesOutput[n->Id]) * (wants[wantsCount] - tmpNodesOutput[n->Id]);
                    ++wantsCount;
                    continue;
                }

                if (ri->first == 0.f) {
                    break;
                }

                calculateNeuronError(n->Id);  // 计算隐藏神经元误差
            }
        }

        // 更新连接权重
        for (auto &connection : nn->Connections) {
//            connection.Weight += Opts.LearnRate * (tmpNodesOutputError[connection.ConnectedNeuronId[1]] * tmpNodesOutput[connection.ConnectedNeuronId[0]] + connection.Weight);
            connection.Weight += Opts.LearnRate * tmpNodesOutputError[connection.ConnectedNeuronId[1]] * tmpNodesOutput[connection.ConnectedNeuronId[0]];
        }

        // 更新神经元偏置
        for (auto &n : nn->Neurons) {
            if (n.Layer != 0) {
                n.Bias += Opts.LearnRate * tmpNodesOutputError[n.Id];
            }
        }

        if (outputs.size() != wants.size()) {
            std::cerr << "BackPropagation outputs.size(" << outputs.size() << ") != wants.size(" << wants.size() << ")\n";
            exit(0);
        }

        return GetPrecision(outputs, wants) >= Opts.FitnessThreshold;
    };

    void Generation::MutateWeightDirect(Connection &c) {
        c.Weight = float(random() % (Opts.WeightRange * 2000000) - Opts.WeightRange * 1000000) / 1000000.f;
    }

    void Generation::MutateWeightNear(Connection &c) {
        float tmpWeight = c.Weight + float(random() % (Opts.MutateWeightNearRange * 2000000) - Opts.MutateWeightNearRange * 1000000) / 1000000.f;
        if (tmpWeight > float(Opts.WeightRange)) {
            c.Weight = float(Opts.WeightRange);
            return;
        }
        if (tmpWeight < -float(Opts.WeightRange)) {
            c.Weight = -float(Opts.WeightRange);
            return;
        }
        c.Weight = tmpWeight;
    }

    void Generation::MutateBiasDirect(Neuron &o) {
        o.Bias = float(random() % (Opts.BiasRange * 2000000) - Opts.BiasRange * 1000000) / 1000000;
    }

    void Generation::MutateBiasNear(Neuron &o) {
        float tmpBias = o.Bias + float(random() % (Opts.MutateBiasNearRange * 2000000) - Opts.MutateBiasNearRange * 1000000) / 1000000.f;
        if (tmpBias > float(Opts.BiasRange)) {
            o.Bias = float(Opts.BiasRange);
            return;
        }
        if (tmpBias < -float(Opts.BiasRange)) {
            o.Bias = -float(Opts.BiasRange);
            return;
        }
        o.Bias = tmpBias;
    }

    void Generation::MutateAddNeuron(NetworkGenome &nn) {
        Connection &choosingConnection = nn.Connections[random() % nn.Connections.size()];  // 选出一条连接

//        if (!choosingConnection.Enable) {
//            return;  // 如果连接被禁用就。。。算了不用这个判断
//        }

        uint nid0 = choosingConnection.ConnectedNeuronId[0];  // 获取连接左边的神经元的id
        uint nid1 = choosingConnection.ConnectedNeuronId[1];  // 获取连接右边的神经元的id

        std::vector<uint> tmpRightNeuronIds;
        for (auto &n : nn.Connections) {
            if (n.ConnectedNeuronId[0] == nid0) {
                tmpRightNeuronIds.push_back(n.ConnectedNeuronId[1]);
            }
        }
        for (auto &n : nn.Connections) {
            if (n.ConnectedNeuronId[1] == nid1) {
                for (auto &tnn : tmpRightNeuronIds) {
                    if (tnn == n.ConnectedNeuronId[0]) {
                        return;  // 如果已经存在相同的连接左右神经元id的神经元，则不继续执行添加此神经元
                    }
                }
            }
        }

        mtx.lock();
        uint newNid = neuralNetwork.HiddenNeuronInnovations.size() + Opts.InputSize + Opts.OutputSize + neuralNetwork.FCHidenNeuronSize;  // 新的神经元id为全部藏神经元数量+输入神经元数量+输出神经元数量+全连接网络隐藏神经元数量(如有)
//        if (HiddenNeuronInnovations.find({nid0, nid1}) == HiddenNeuronInnovations.end()) {  // 从全部隐藏神经元innovMap里面查看是否存在相同位置的神经元
        if (!neuralNetwork.HiddenNeuronInnovations.contains({nid0, nid1})) {  // 从全部隐藏神经元innovMap里面查看是否存在相同位置的神经元
            neuralNetwork.HiddenNeuronInnovations[{nid0, nid1}] = newNid;  // 如果不存在则新增记录插入连接左右两个神经元id对应的隐藏层神经元id
        } else {
            newNid = neuralNetwork.HiddenNeuronInnovations[{nid0, nid1}];  // 如果存在则使用已有隐藏层神经元id
        }
        mtx.unlock();

        std::map<uint, Neuron *> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址
        for (auto &o : nn.Neurons) {
//            if (tmpNeuronMap.find(o.Id) == tmpNeuronMap.end()) {
            tmpNeuronMap[o.Id] = &o;
//            }
        }

        Neuron newNeuron = {
                .Id = newNid,
                .Bias = float(random() % (Opts.BiasRange * 2000000) - Opts.BiasRange * 1000000) / 1000000.f,
                .Layer = (tmpNeuronMap[nid0]->Layer + tmpNeuronMap[nid1]->Layer) / 2.f,
        };
        nn.Neurons.push_back(newNeuron);

        choosingConnection.Enable = false;  // 禁用此条连接

        Connection newConn0 = {  // 添加左侧连接
                .ConnectedNeuronId = {
                        nid0,
                        newNid,
                },
                .Weight = float(random() % (Opts.WeightRange * 2000000) - Opts.WeightRange * 1000000) / 1000000.f,
                .Enable = true,
        };
        nn.Connections.push_back(newConn0);

        Connection newConn1 = {  // 添加右侧连接
                .ConnectedNeuronId = {
                        newNid,
                        nid1,
                },
                .Weight = float(random() % (Opts.WeightRange * 2000000) - Opts.WeightRange * 1000000) / 1000000.f,
                .Enable = true,
        };
        nn.Connections.push_back(newConn1);

//        mtx.lock();
//        ConnectionInnovations[{nid0, newNid}] = ConnectionInnovations.size(); // 记录老的左侧神经元id和新神经元id对应的新连接innov
//        ConnectionInnovations[{newNid, nid1}] = ConnectionInnovations.size(); // 记录老的新神经元id和右侧神经元id对应的新连接innov
//        mtx.unlock();
    }

    void Generation::MutateAddConnection(NetworkGenome &nn) {
        Neuron &choosingNeuron0 = nn.Neurons[random() % nn.Neurons.size()];  // 选出第一个神经元
        Neuron &choosingNeuron1 = nn.Neurons[random() % nn.Neurons.size()];  // 选出第二个神经元

        if (choosingNeuron0.Layer == choosingNeuron1.Layer) {  // 同一layer包含 choosingNeuron0.Id == choosingNeuron1.Id
            return;
        }

        uint nid0 = choosingNeuron0.Id;
        uint nid1 = choosingNeuron1.Id;

        if (choosingNeuron0.Layer > choosingNeuron1.Layer) { // 保持从左到右编号
            nid0 = choosingNeuron1.Id;
            nid1 = choosingNeuron0.Id;
        }

        for (auto &c : nn.Connections) {
            if (c.ConnectedNeuronId[0] == nid0 && c.ConnectedNeuronId[1] == nid1) {
                return; // 已存在这个链接
            }
        }

        Connection newConn = {  // 添加连接
                .ConnectedNeuronId = {
                        nid0,
                        nid1,
                },
                .Weight = float(random() % (Opts.WeightRange * 2000000) - Opts.WeightRange * 1000000) / 1000000.f,
                .Enable = true,
        };
        nn.Connections.push_back(newConn);

//        mtx.lock();
//        if (ConnectionInnovations.find({nid0, nid1}) == ConnectionInnovations.end()) {  // 查询所有连接的innov是否存在
//            ConnectionInnovations[{nid0, nid1}] = ConnectionInnovations.size();  // 如果不存在则新增连接的innov
//        }
//        mtx.unlock();
    }

    void Generation::MutateEnableConnection(NetworkGenome &nn) {
        Connection &choosingConnection = nn.Connections[random() % nn.Connections.size()];  // 选出一条连接
        if (choosingConnection.Enable) {
            choosingConnection.Enable = false;
            return;
        }
        choosingConnection.Enable = true;
    }

    void Generation::EnableAllConnections(NetworkGenome &nn) {
        for (auto &c : nn.Connections) {
            if (!c.Enable) {
                c.Enable = true;
            }
        }
    }

    void Generation::MutateNetworkGenome(NetworkGenome &nn) {
        for (auto &c : nn.Connections) {
            if (float(random() % 1000) / 1000.f < Opts.MutateWeightRate) {
                if (float(random() % 1000) / 1000.f < Opts.MutateWeightDirectOrNear) {
                    MutateWeightDirect(c);
                } else {
                    MutateWeightNear(c);
                }
            }
        }

        for (auto &n : nn.Neurons) {
            if (float(random() % 1000) / 1000.f < Opts.MutateBiasRate && n.Layer > 0.f) {
                if (float(random() % 1000) / 1000.f < Opts.MutateBiasDirectOrNear) {
                    MutateBiasDirect(n);
                } else {
                    MutateBiasNear(n);
                }
            }
        }

        if (float(random() % 1000) / 1000.f < Opts.MutateAddNeuronRate) {
            MutateAddNeuron(nn);
        }

        if (float(random() % 1000) / 1000.f < Opts.MutateAddConnectionRate) {
            MutateAddConnection(nn);
        }

        if (float(random() % 1000) / 1000.f < Opts.MutateEnableConnectionRate) {
            MutateEnableConnection(nn);
        }
    }

    NetworkGenome Generation::GetChildByCrossing(NetworkGenome *nn0, NetworkGenome *nn1) {
        if (float(random() % 1000) / 1000.f > Opts.CrossoverRate || nn0 == nn1) {
            return *nn1; // nn0 是冠军中的个体， nn1 是剩余的，冠军已经保留了原始基因，所以按照概率保留非冠军基因
        }

        std::vector<Connection> newConnections;// 记录全部涉及的连接

        for (auto c0 : nn0->Connections) {  // 遍历第一个神经元的所有连接
            bool isThisConnectionExists = false;
            for (auto c1 : nn1->Connections) {// 遍历第二个神经元的所有连接
                if (c0.ConnectedNeuronId == c1.ConnectedNeuronId) {
                    isThisConnectionExists = true;
                    if (random() % 2 == 0) {  // 若有相同对应神经元id的连接，则随机选一条
                        newConnections.push_back(c0);
                    } else {
                        newConnections.push_back(c1);
                    }
                    break;
                }
            }
            if (!isThisConnectionExists) {
                newConnections.push_back(c0);// 无相同对应神经元id的连接，则选它
            }
//            remainingIds.insert(remainingIds.end(), c0.ConnectedNeuronId.begin(), c0.ConnectedNeuronId.end());
        }

        for (auto c1 : nn1->Connections) {// 遍历第二个神经元的所有连接
            bool isThisConnectionExists = false;
            for (auto c2 : newConnections) {// 遍历根据第一个连接加上第二个神经元中第一个神经元没有的连接
                if (c1.ConnectedNeuronId == c2.ConnectedNeuronId) {
                    isThisConnectionExists = true;
                    break;
                }
            }
            if (!isThisConnectionExists) {
                newConnections.push_back(c1);// 无相同对应神经元id的连接，则选它
//                remainingIds.insert(remainingIds.end(), c1.ConnectedNeuronId.begin(), c1.ConnectedNeuronId.end());
            }
        } // 两个for循环选出全部连接

        std::map<uint, uint> remainingIds;  // 记录所有涉及的神经元id
        std::map<uint, Neuron *> tmpNeuron0Map;

        for (auto &n : nn0->Neurons) {
            tmpNeuron0Map[n.Id] = &n;
            remainingIds[n.Id] = 0;
        }

        std::map<uint, Neuron *> tmpNeuron1Map;
        for (auto &n : nn1->Neurons) {
            tmpNeuron1Map[n.Id] = &n;
            remainingIds[n.Id] = 0;
        }

        std::vector<Neuron> newNeurons;

        for (auto &i : remainingIds) {
//            bool isNeuronIn0 = (tmpNeuron0Map.find(i.first) != tmpNeuron0Map.end());
            bool isNeuronIn0 = (tmpNeuron0Map.contains(i.first));
//            bool isNeuronIn1 = (tmpNeuron1Map.find(i.first) != tmpNeuron1Map.end());
            bool isNeuronIn1 = (tmpNeuron1Map.contains(i.first));

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

        return NetworkGenome{
                .Neurons = newNeurons,
                .Connections = newConnections,
        };
    }
}

#endif //MYNEAT_GENERATION_HPP
