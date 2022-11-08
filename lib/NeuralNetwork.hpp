#pragma once

#ifndef MYNEAT_NERAULNETWORK_HPP
#define MYNEAT_NERAULNETWORK_HPP

#include <vector>
#include <map>
#include <array>
#include <functional>
#include <fstream>
#include "raylib.h"
#include "Option.hpp"

namespace znn {
    struct Neuron {
        uint Id;
        float Bias;
        float Layer;
    };

    struct Connection {
        std::array<uint, 2> ConnectedNeuronId;
        float Weight;
        bool Enable;
    };

    struct NetworkGenome {
        std::vector<Neuron> Neurons;
        std::vector<Connection> Connections;
    };

    class NeuralNetwork {
    public:
        std::map<std::array<uint, 2>, uint> HiddenNeuronInnovations;  // 只记录插入连接左右两个神经元id对应的隐藏层神经元id，新增神经元变异的时候全部个体需要检查唯一性，使用时必须使用mutex
        int FCHidenNeuronSize = 0;

        NetworkGenome NewNN();

        NetworkGenome NewFCNN(std::vector<int> hideLayers);

        NetworkGenome SimplifyRemoveUselessConnectionRight(NetworkGenome nn);

        NetworkGenome SimplifyRemoveUselessConnectionLeft(NetworkGenome nn);

        NetworkGenome SimplifyRemoveDisable(NetworkGenome nn);

        std::vector<float> FeedForwardPredict(NetworkGenome *nn, std::vector<float> inputs);

        void ExportNNToDot(NetworkGenome &nn, std::string fileName);

        void ExportNN(NetworkGenome &nn, std::string fileName);

        NetworkGenome ImportNN(std::string fileName);

        void ExportInnovations(std::string fileName);

        void ImportInnovations(std::string fileName);
    };

    NetworkGenome NeuralNetwork::NewNN() {
        if (Opts.InputSize <= 0 || Opts.OutputSize <= 0) {
            std::cerr << "Input or Output size fault: Input " << Opts.InputSize << ", Output " << Opts.OutputSize << std::endl;
            exit(0);
        }

        std::vector<Neuron> newNeurons;
        std::vector<Connection> newConnections;

        for (uint i = 0; i < Opts.InputSize; ++i) {
            Neuron tmpNeuron = {
                    .Id = i,
                    .Bias = 0.f,
                    .Layer = 0.f,
            };
            newNeurons.push_back(tmpNeuron);
        }

        for (uint i = 0; i < Opts.OutputSize; ++i) {
            uint id = i + Opts.InputSize;
            Neuron tmpNeuron = {
                    .Id = id,
                    //                    .Bias = 1.f,
                    .Bias = float(random() % (Opts.BiasRange * 200) - Opts.BiasRange * 100) / 100,
                    .Layer = 1.f,
            };
            newNeurons.push_back(tmpNeuron);

            for (auto &n : newNeurons) {
                if (n.Layer == 0.f) {
                    Connection tmpConnection = {
                            .ConnectedNeuronId= {n.Id, id},
                            //                            .Weight = 1.f,
                            .Weight = float(random() % (Opts.WeightRange * 200) - Opts.WeightRange * 100) / 100,
                            .Enable = true,
                    };
                    newConnections.push_back(tmpConnection);
                    //                    if (ConnectionInnovations.find({n.Id, tmpNeuron.Id}) == ConnectionInnovations.end()) {
                    //                        ConnectionInnovations[{n.Id, tmpNeuron.Id}] = ConnectionInnovations.size();
                    //                    }
                }
            }
        }

        return NetworkGenome{
                .Neurons = newNeurons,
                .Connections = newConnections,
        };
    }

    NetworkGenome NeuralNetwork::NewFCNN(std::vector<int> hideLayers) {  // 固定神经网络，输入隐藏层及对应神经元数量数
        if (Opts.InputSize <= 0 || Opts.OutputSize <= 0) {
            std::cerr << "Input or Output size fault: Input " << Opts.InputSize << ", Output " << Opts.OutputSize << std::endl;
            exit(0);
        }

        if (FCHidenNeuronSize == 0) {
            for (int &l : hideLayers) {
                FCHidenNeuronSize += l;
            }
        }

        std::vector<Neuron> newNeurons;
        std::vector<Connection> newConnections;

        for (uint i = 0; i < Opts.InputSize; ++i) {
            Neuron tmpNeuron = {
                    .Id = i,
                    .Bias = 1.f,
                    .Layer = 0.f,
            };
            newNeurons.push_back(tmpNeuron);
        }

        for (uint i = 0; i < Opts.OutputSize; ++i) {
            uint id = i + Opts.InputSize;
            Neuron tmpNeuron = {
                    .Id = id,
                    .Bias = float(random() % (Opts.BiasRange * 200) - Opts.BiasRange * 100) / 100,
                    .Layer = 1.f,
            };
            newNeurons.push_back(tmpNeuron);
        }

        float layerStep = 1.f / float(hideLayers.size() + 1);
        float thisLayer = 0.f;
        uint id = Opts.InputSize + Opts.OutputSize;

        for (int &l : hideLayers) {
            thisLayer += layerStep;
            for (uint i = 0; i < l; ++i) {
                Neuron tmpNeuron = {
                        .Id = id,
                        .Bias = float(random() % (Opts.BiasRange * 200) - Opts.BiasRange * 100) / 100,
                        .Layer = thisLayer,
                };
                newNeurons.push_back(tmpNeuron);
                ++id;
            }
        }

        std::map<float, std::vector<Neuron *>> tmpLayerMap;  // 记录全部层，因为记录的是神经元地址，需要的时候才能临时生成记录
        float lastLayer = 0.f;
        thisLayer = 0.f;

        for (auto &n : newNeurons) {
            tmpLayerMap[n.Layer].push_back(&n);
        }

        for (auto &t : tmpLayerMap) {
            if (t.first != 0.f) {
                for (auto &n : t.second) {
                    for (auto &ln : tmpLayerMap[lastLayer]) {
                        Connection tmpConnection = {
                                .ConnectedNeuronId= {ln->Id, n->Id},
                                .Weight = float(random() % (Opts.WeightRange * 200) - Opts.WeightRange * 100) / 100,
                                .Enable = true,
                        };
                        newConnections.push_back(tmpConnection);
                    }
                }
            }
            lastLayer = thisLayer;
            thisLayer += layerStep;
        }

        return NetworkGenome{
                .Neurons = newNeurons,
                .Connections = newConnections,
        };
    }

    NetworkGenome NeuralNetwork::SimplifyRemoveUselessConnectionRight(NetworkGenome nn) { //合并中途凭空出现节点到右边的连接
        std::map<uint, std::vector<Connection *>> remainingLeftIds;
        std::map<uint, std::vector<Connection *>> remainingRightIds;
        std::map<uint, std::vector<Connection *>> removeIds;
        std::map<uint, Neuron> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址

        for (auto &n : nn.Neurons) {
            tmpNeuronMap[n.Id] = n;
        }

        for (;;) {
            remainingLeftIds.clear();
            remainingRightIds.clear();
            removeIds.clear();

            for (auto &n : nn.Connections) {
                remainingRightIds[n.ConnectedNeuronId[1]].push_back(&n);   // 添加所有连接右边的神经元id
            }

            for (auto &n : nn.Connections) {
                if (remainingRightIds.find(n.ConnectedNeuronId[0]) != remainingRightIds.end() || tmpNeuronMap[n.ConnectedNeuronId[0]].Layer == 0.f) {  // c++17
                    //if (remainingRightIds.contains(n.ConnectedNeuronId[0]) || tmpNeuronMap[n.ConnectedNeuronId[0]].Layer == 0.f) {   // c++20
                    remainingLeftIds[n.ConnectedNeuronId[0]].push_back(&n);  // 添加所左边的神经元id（可作为另外一条连接的右边神经元）(左边还有连接)
                } else {
                    removeIds[n.ConnectedNeuronId[0]].push_back(&n);
                }
            }

            for (auto &r : removeIds) {
                float thisBias = tmpNeuronMap[r.first].Bias;
                for (auto &cp : r.second) {
                    tmpNeuronMap[cp->ConnectedNeuronId[1]].Bias += Opts.ActiveFunction(thisBias) * cp->Weight;
                }
            }

            std::vector<Neuron> newNeurons;
            std::vector<Connection> newConnections;

            for (auto &r : remainingLeftIds) {
                newNeurons.push_back(tmpNeuronMap[r.first]);
                for (auto &cp : r.second) {
                    newConnections.push_back(*cp);
                }
            }

            nn = {
                    .Neurons=newNeurons,
                    .Connections=newConnections,
            };

            if (removeIds.empty()) {
                break;
            }
        }

        for (auto &n : tmpNeuronMap) {
            if ((remainingLeftIds.find(n.first) == remainingLeftIds.end() && n.second.Layer == 0.f) || (remainingRightIds.find(n.first) == remainingRightIds.end() && n.second.Layer == 1.f) ||
                (n.second.Layer == 1.f)) {
                //            if ((!remainingLeftIds.contains(n.first) && n.second.Layer == 0.f) || (remainingRightIds.contains(n.first) && n.second.Layer == 1.f) || (n.second.Layer == 1.f)) {
                nn.Neurons.push_back(n.second);
            }
        }

        return nn;
    }

    NetworkGenome NeuralNetwork::SimplifyRemoveUselessConnectionLeft(NetworkGenome nn) { // 实现从左到右的无效连接移除
        std::map<uint, std::vector<Connection *>> remainingLeftIds;
        std::map<uint, std::vector<Connection *>> remainingRightIds;
        std::map<uint, Neuron> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址

        for (auto &n : nn.Neurons) {
            tmpNeuronMap[n.Id] = n;
        }

        for (;;) {
            bool isFinished = true;
            remainingLeftIds.clear();
            remainingRightIds.clear();

            for (auto &n : nn.Connections) {
                remainingLeftIds[n.ConnectedNeuronId[0]].push_back(&n);   // 添加所有连接左边的神经元id
            }


            for (auto &n : nn.Connections) {
                if (remainingLeftIds.find(n.ConnectedNeuronId[1]) != remainingLeftIds.end() || tmpNeuronMap[n.ConnectedNeuronId[1]].Layer == 1.f) {
                    //                if (remainingLeftIds.contains(n.ConnectedNeuronId[1]) || tmpNeuronMap[n.ConnectedNeuronId[1]].Layer == 1.f) {
                    remainingRightIds[n.ConnectedNeuronId[1]].push_back(&n);  // 添加所右边的神经元id（可作为另外一条连接的左边神经元）(右边还有连接)
                } else {
                    isFinished = false;
                }
            }

            std::vector<Neuron> newNeurons;
            std::vector<Connection> newConnections;

            for (auto &r : remainingRightIds) {
                newNeurons.push_back(tmpNeuronMap[r.first]);
                for (auto &cp : r.second) {
                    newConnections.push_back(*cp);
                }
            }

            nn = {
                    .Neurons=newNeurons,
                    .Connections=newConnections,
            };

            if (isFinished) {
                break;
            }
        }

        for (auto &n : tmpNeuronMap) {
            if ((remainingLeftIds.find(n.first) != remainingLeftIds.end() && remainingRightIds.find(n.first) == remainingRightIds.end()) || (n.second.Layer == 0.f) ||
                (remainingRightIds.find(n.first) == remainingRightIds.end() && n.second.Layer == 1.f)) {
                //            if ((remainingLeftIds.contains(n.first) && !remainingRightIds.contains(n.first)) || (n.second.Layer == 0.f) || (!remainingRightIds.contains(n.first) && n.second.Layer == 1.f)) {
                nn.Neurons.push_back(n.second);
            }
        }

        return nn;
    }

    NetworkGenome NeuralNetwork::SimplifyRemoveDisable(NetworkGenome nn) {
        std::vector<Connection> newConnections;
        std::map<uint, uint> remainingIds;

        for (auto c : nn.Connections) {
            if (c.Enable && std::abs(c.Weight) > 0.001f) {  // Weight绝对值小于0.001算成disable
                newConnections.push_back(c);
                remainingIds[c.ConnectedNeuronId[0]] = 0;
                remainingIds[c.ConnectedNeuronId[1]] = 0;
            }
        }

        std::map<uint, Neuron *> tmpNeuronMap;

        for (auto &n : nn.Neurons) {
            tmpNeuronMap[n.Id] = &n;
            if (n.Layer == 0.f || n.Layer == 1.f) {
                remainingIds[n.Id] = 0;
            }
        }

        std::vector<Neuron> newNeurons;

        for (auto &i : remainingIds) {
            newNeurons.push_back(*tmpNeuronMap[i.first]);
        }

        return NetworkGenome{
                .Neurons = newNeurons,
                .Connections = newConnections,
        };
    }

    std::vector<float> NeuralNetwork::FeedForwardPredict(NetworkGenome *nn, std::vector<float> inputs) {
        std::map<uint, Neuron *> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址
        std::map<float, std::vector<Neuron *>> tmpLayerMap;  // 记录层对应神经元，同上因为记录的是神经元地址，需要的时候才能临时生成记录

        for (auto &n : nn->Neurons) {
            //            if (tmpNeuronMap.find(n.Id) == tmpNeuronMap.end()) {
            tmpNeuronMap[n.Id] = &n;
            //            }
            //            if (n.NNLayer > 0.f && n.NNLayer < 1.f && tmpLayerMap.find(n.NNLayer) == tmpLayerMap.end()) {
            tmpLayerMap[n.Layer].push_back(&n);
        }


        if (Opts.InputSize != inputs.size()) {
            std::cerr << "FeedForwardPredict: Input length " << inputs.size() << " diffrent with NN input nodes " << Opts.InputSize << std::endl;
            std::exit(0);
        }

        std::map<uint, float> tmpNodesOutput;

        //        for (Neuron &n : nn.Neurons) { // 初始化输入节点
        //            if (n.NNLayer == 0.f) {
        //                tmpNodesOutput[n.Id] = inputs[n.Id];
        //            }
        //        }

        std::function<void(uint)> calculateNeuron = [&](uint nid) {
            tmpNodesOutput[nid] = 0.f;
            for (auto &connection : nn->Connections) {
                if (connection.ConnectedNeuronId[1] == nid && connection.Enable) {
                    tmpNodesOutput[nid] += tmpNodesOutput[connection.ConnectedNeuronId[0]] * connection.Weight;
                }
            }
            tmpNodesOutput[nid] = Opts.ActiveFunction(tmpNodesOutput[nid] + tmpNeuronMap[nid]->Bias);
        };

        std::vector<float> outputs;

        //        if (!tmpLayerMap.empty()) {
        for (auto &l : tmpLayerMap) {    // 神经元根据layer排序
            for (auto &n : l.second) {
                if (l.first == 0.f) {   // 初始化输入节点
                    tmpNodesOutput[n->Id] = inputs[n->Id];
                    continue;
                }

                //                if (tmpNeuronMap.find(n->Id) != tmpNeuronMap.end()) {   // 从本次临时神经元id对应的神经元记录中查询，确保id存在，避免多个神经网络混淆
                calculateNeuron(n->Id);  // 计算隐藏神经元
                //                    }

                if (l.first == 1.f) {  // 计算输出神经元
                    outputs.push_back(tmpNodesOutput[n->Id]);
                }
            }
        }
        //        }

        //        for (uint i = 0; i < Opts.OutputSize; ++i) {  // 计算输出神经元
        //            uint id = i + Opts.InputSize;
        //            calculateNeuron(id);
        //            outputs.push_back(tmpNodesOutput[id]);
        //        }

        return outputs;
    };

    void NeuralNetwork::ExportNNToDot(NetworkGenome &nn, std::string fileName) {
        std::string data = "digraph G {\n    rankdir=\"LR\";\n";
        uint inId = 0;
        uint outId = 0;

        for (auto &n : nn.Neurons) {
            std::string line;
            std::stringstream streamBias;
            if (n.Layer == 0.f) {
                line = "    subgraph cluster0{" + std::to_string(n.Id) + " [fontsize=24,width=0,height=0,color=lightblue,style=filled,shape=component,width=1,height=1,label=\"Input_" +
                       std::to_string(inId) + "\"]}\n";
                ++inId;
            } else if (n.Layer == 1.f) {
                streamBias << std::setprecision(3) << n.Bias;
                line = "    subgraph cluster1{" + std::to_string(n.Id) + " [fontsize=24,width=0,height=0,color=lightgray,style=filled,shape=diamond,width=1,height=1,label=\"Output_" +
                       std::to_string(outId) + "\\n(" + streamBias.str() + ")\"]}\n";
                ++outId;
            } else {
                streamBias << std::setprecision(3) << n.Bias;
                //                line = "    subgraph cluster" + std::to_string(int(n.Layer * float(1000000000))) + "{" + std::to_string(n.Id) + " [fontsize=27,width=0,height=0,color=lightgreen,style=filled,shape=box,width=1,height=1,label=\"(" + streamBias.str() + ")\"]}\n";
                line = "    " + std::to_string(n.Id) + " [fontsize=27,width=0,height=0,color=lightgreen,style=filled,shape=box,width=1,height=1,label=\"(" + streamBias.str() + ")\"]\n";
            }
            data += line;
        }

        for (auto &c : nn.Connections) {
            std::stringstream streamWeight;
            streamWeight << std::setprecision(3) << c.Weight;
            data += "    " + std::to_string(c.ConnectedNeuronId[0]) + " -> " + std::to_string(c.ConnectedNeuronId[1]) + " [fontsize=27,label=\"" + streamWeight.str() + "\",decorate = true]\n";
        }

        data += "}";
        std::ofstream file(fileName + ".dot");
        file << data;
        file.close();
    }

    void NeuralNetwork::ExportNN(NetworkGenome &nn, std::string fileName) {
        std::string data = std::to_string(Opts.InputSize) + "," + std::to_string(Opts.OutputSize) + "\n~\n";

        for (auto &n : nn.Neurons) {
            std::stringstream streamLayer;
            streamLayer << std::scientific << n.Layer;
            std::stringstream streamBias;
            streamBias << std::scientific << n.Bias;
            data += std::to_string(n.Id) + "," + streamLayer.str() + "," + streamBias.str() + "\n";
        }

        data += "~\n";

        for (auto &c : nn.Connections) {
            std::stringstream streamWeight;
            streamWeight << std::scientific << c.Weight;
            data += std::to_string(c.ConnectedNeuronId[0]) + "->" + std::to_string(c.ConnectedNeuronId[1]) + "," + streamWeight.str() + "\n";
        }

        std::ofstream file(fileName + ".nn");
        file << data;
        file.close();
    }

    NetworkGenome NeuralNetwork::ImportNN(std::string fileName) {
        std::string line;
        std::ifstream input_file(fileName + ".nn");

        if (!input_file.is_open()) {
            std::cerr << "Could not open the file - '" << fileName << "'\n";
            exit(0);
        }

        uint InputSize = 0;
        uint OutputSize = 0;

        std::vector<Neuron> newNeurons;
        std::vector<Connection> newConnections;
        std::map<uint, uint> tmpIdMap;

        int dataType = 0;
        while (getline(input_file, line)) {
            if (line == "~") {
                ++dataType;
                continue;
            }

            if (dataType == 0) {
                auto datas = SplitString(line, ",");
                InputSize = uint(std::stoi(datas[0]));
                OutputSize = uint(std::stoi(datas[1]));
                continue;
            }

            if (dataType == 1) {
                auto datas = SplitString(line, ",");
                float nnLayer = std::stof(datas[1]);
                uint id = uint(std::stoi(datas[0]));
                newNeurons.push_back(Neuron{
                        .Id = id,
                        .Bias = std::stof(datas[2]),
                        .Layer = nnLayer,
                });
                tmpIdMap[id] = 0;
                continue;
            }

            auto datas = SplitString(line, ",");
            auto cnid = SplitString(datas[0], "->");
            newConnections.push_back(Connection{
                    .ConnectedNeuronId = {uint(std::stoi(cnid[0])), uint(std::stoi(cnid[1]))},
                    .Weight = std::stof(datas[1]),
                    .Enable = true,
            });
        }

        input_file.close();

        if (HiddenNeuronInnovations.empty() && (Opts.InputSize > InputSize || Opts.OutputSize > OutputSize)) {
            std::cerr << "HiddenNeuronInnovations is empty, import .innov file fisrt.";
            exit(0);
        }

        uint inInnovMapSize = 0;
        for (auto &n : HiddenNeuronInnovations) {
            //            if (tmpIdMap.contains(n.second)) {
            if (tmpIdMap.find(n.second) != tmpIdMap.end()) {
                ++inInnovMapSize;
            }
        }

        FCHidenNeuronSize = newNeurons.size() - InputSize - OutputSize - inInnovMapSize;

        if (Opts.InputSize > InputSize) {
            for (uint i = 0; i < Opts.InputSize - InputSize; ++i) {
                newNeurons.push_back(Neuron{
                        .Id = uint(HiddenNeuronInnovations.size()) + OutputSize + InputSize + i + FCHidenNeuronSize,
                        //                        .Bias = 1.f,
                        .Bias = float(random() % (Opts.BiasRange * 200) - Opts.BiasRange * 100) / 100,
                        .Layer = 0.f,
                });
            }
        }

        if (Opts.OutputSize > OutputSize) {
            for (uint i = 0; i < Opts.OutputSize - OutputSize; ++i) {
                newNeurons.push_back(Neuron{
                        .Id = uint(HiddenNeuronInnovations.size()) + OutputSize + i + Opts.InputSize + FCHidenNeuronSize,
                        //                        .Bias = 1.f,
                        .Bias = float(random() % (Opts.BiasRange * 200) - Opts.BiasRange * 100) / 100,
                        .Layer = 1.f,
                });
            }
        }

        return NetworkGenome{
                .Neurons = newNeurons,
                .Connections = newConnections,
        };
    }

    void NeuralNetwork::ExportInnovations(std::string fileName) {
        std::string data;

        for (auto &i : HiddenNeuronInnovations) {
            data += std::to_string(i.first[0]) + "," + std::to_string(i.first[1]) + "," + std::to_string(i.second) + "\n";
        }
        std::ofstream file(fileName + ".innov");
        file << data;
        file.close();
    }

    void NeuralNetwork::ImportInnovations(std::string fileName) {
        std::string line;
        std::ifstream input_file(fileName + ".innov");

        if (!input_file.is_open()) {
            std::cerr << "Could not open the file - '" << fileName << "'\n";
            exit(0);
        }

        while (getline(input_file, line)) {
            auto datas = SplitString(line, ",");
            HiddenNeuronInnovations[{uint(std::stoi(datas[0])), uint(std::stoi(datas[1]))}] = uint(std::stoi(datas[2]));
        }
    }

    struct lineInfo {
        uint IdA;
        uint IdB;
        float R;
        Color C;
    };

    std::map<uint, Vector3> NodeId2Pos;
    std::map<uint, Color> NodId2Color;
    std::vector<lineInfo> connectedNodesInfo;

    void show3dNN() {
        const int screenWidth = 1280;
        const int screenHeight = 720;

        SetConfigFlags(FLAG_MSAA_4X_HINT);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);

        InitWindow(screenWidth, screenHeight, "SimpleNEAT NN");

        // Define the camera to look into our 3d world
        Camera3D camera = {0};
        camera.position = (Vector3) {0.0f, 0.0f, 12.0f}; // Camera position
        camera.target = (Vector3) {0.0f, 0.0f, 0.0f};      // Camera looking at point
        camera.up = (Vector3) {0.0f, 1.0f, 0.0f};          // Camera up vector (rotation towards target)
        camera.fovy = 60.0f;                                // Camera field-of-view Y
        camera.projection = CAMERA_PERSPECTIVE;                   // Camera mode type

        SetCameraMode(camera, CAMERA_FREE); // Set a free camera mode
        SetCameraAltControl(KEY_LEFT_SHIFT);
        SetCameraPanControl(MOUSE_BUTTON_RIGHT);

        SetTargetFPS(60);                   // Set our game to run at 60 frames-per-second

        srandom((unsigned) clock());

        // Main game loop
        while (!WindowShouldClose()) {     // Detect window close button or ESC key
            // Update
            //----------------------------------------------------------------------------------
            UpdateCamera(&camera);

            if (IsKeyDown('Z')) camera.target = (Vector3) {0.0f, 0.0f, 0.0f};
            //----------------------------------------------------------------------------------

            // Draw
            //----------------------------------------------------------------------------------
            BeginDrawing();
            ClearBackground(BLACK);

            BeginMode3D(camera);

            //                DrawCubeV({1.f, 0.f, 0.f}, {0.03f, 0.03f, 0.03f}, RED);
            //                DrawCubeV({0.f, 1.f, 0.f}, {0.03f, 0.03f, 0.03f}, YELLOW);
            //                DrawCubeV({0.f, 0.f, 1.f}, {0.03f, 0.03f, 0.03f}, BLUE);
            //
            //                DrawLine3D({-1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, RED);
            //                DrawLine3D({0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}, YELLOW);
            //                DrawLine3D({0.f, 0.f, -1.f}, {0.f, 0.f, 1.f}, BLUE);

            for (auto &n : NodeId2Pos) {
                DrawCubeV(n.second, {0.1f, 0.1f, 0.1f}, NodId2Color[n.first]);
            }

            for (auto &c : connectedNodesInfo) {
                //                DrawLine3D(NodeId2Pos[c.IdA], NodeId2Pos[c.IdB], c.C);
                DrawCylinderEx(NodeId2Pos[c.IdA], NodeId2Pos[c.IdB], c.R, c.R, 3, c.C);
            }

            EndMode3D();

            DrawFPS(10, 10);
            EndDrawing();
            //----------------------------------------------------------------------------------
        }

        // De-Initialization
        //--------------------------------------------------------------------------------------
        CloseWindow();        // Close window and OpenGL context
        //--------------------------------------------------------------------------------------
    }

    bool update3dLock = false;
    NetworkGenome last3dNN;

    bool isLast3dNN(NetworkGenome &NN) {
        if (NN.Neurons.size() != last3dNN.Neurons.size()) {
            last3dNN = NN;
            return false;
        }

        for (uint i = 0; i < NN.Neurons.size(); ++i) {
            if (NN.Neurons[i].Id != last3dNN.Neurons[i].Id) {
                last3dNN = NN;
                return false;
            }
        }

        return true;
    };

    void update3dNN(NetworkGenome NN) {
        mtx.lock();
        if (update3dLock) {
            mtx.unlock();
        } else {
            update3dLock = true;
            mtx.unlock();

            if (!isLast3dNN(NN)) {
                std::map<float, std::vector<uint>> layer2Ids;

                for (auto &n : NN.Neurons) {
                    layer2Ids[n.Layer].push_back(n.Id);
                }

                float setZyInterval = 1.f;
//                    float setXInterval = 8.f / float(layer2Ids.size());
                float setXInterval = 1.f;
                float layerCount = 0;

                NodeId2Pos.clear();

                for (auto &l2i : layer2Ids) {
                    int rows = int(std::sqrt(float(l2i.second.size())));
                    int columns = int(l2i.second.size() / rows);
                    float startY = -float(rows - 1) * setZyInterval / 2.f;
                    float thisY = startY;
                    float startZ0 = -float(columns) * setZyInterval / 2.f;
                    float startZ1 = -float(columns - 1) * setZyInterval / 2.f;
                    float thisZ;
                    int reMainColumns = l2i.second.size() % rows;
                    int row = 0;

                    for (uint i = 0; i < l2i.second.size(); ++i) {
                        if (l2i.first == 0.f) {
                            NodId2Color[l2i.second[i]] = BLUE;
                        } else if (l2i.first == 1.f) {
                            NodId2Color[l2i.second[i]] = RED;
                        } else {
                            NodId2Color[l2i.second[i]] = YELLOW;
                        }

                        if (i % rows < reMainColumns && l2i.second.size() % rows != 0) {
                            thisZ = startZ0 + setZyInterval * float(row);
                        } else {
                            thisZ = startZ1 + setZyInterval * float(row);
                        }

                        NodeId2Pos[l2i.second[i]] = {-(float(layer2Ids.size() - 1) * setXInterval / 2.f + (float(random() % 30) / 100.f - 0.15f) * setXInterval) + setXInterval * layerCount,
                                                     thisY + (float(random() % 30) / 100.f - 0.15f) * setZyInterval, thisZ + (float(random() % 30) / 100.f - 0.15f) * setZyInterval};
                        thisY += setZyInterval;

                        if ((i + 1) % rows == 0) {
                            thisY = startY;
                            ++row;
                        }
                    }
                    ++layerCount;
                }
            }

            connectedNodesInfo.clear();

            for (auto &conn : NN.Connections) {
                if (conn.Enable) {
                    if (conn.Weight > 0) {
                        connectedNodesInfo.push_back(lineInfo{conn.ConnectedNeuronId[0], conn.ConnectedNeuronId[1], conn.Weight * 0.0015f + 0.0001f, ColorAlpha(RED, 0.5f)});
                    } else {
                        connectedNodesInfo.push_back(lineInfo{conn.ConnectedNeuronId[0], conn.ConnectedNeuronId[1], -conn.Weight * 0.0015f + 0.0001f, ColorAlpha(BLUE, 0.5f)});
                    }
                    //            } else {
                    //                connectedNodesInfo.push_back(lineInfo{conn.ConnectedNeuronId[0], conn.ConnectedNeuronId[1], 0.0001f, ColorAlpha(GRAY, 0.3f)});
                }
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));

            update3dLock = false;
        }
    }
}

#endif //MYNEAT_NERAULNETWORK_HPP

