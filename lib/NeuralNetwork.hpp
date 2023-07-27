#pragma once

#ifndef MYNEAT_NERAULNETWORK_HPP
#define MYNEAT_NERAULNETWORK_HPP

#include <vector>
#include <map>
#include <array>
#include <functional>
#include <fstream>
#include "Option.hpp"

#ifndef NO_3DNN

#include "raylib.h"

#endif

namespace znn {
    struct Neuron {  // 定义神经元结构体
        ulong Id;  // 神经元ID
        float Bias;  // 神经元偏置
        double Layer;  // 神经元的逻辑位置（层）
    };

    struct Connection {  // 定义神经连接结构体
        std::array<ulong, 2> ConnectedNeuronId;  // 神经连接由两个神经元而定
        float Weight; // 连接的权重
        bool Enable; // 连接是否启用
    };

    struct NetworkGenome {
        uint Age = 0;
        std::vector<Neuron> Neurons;
        std::vector<Connection> Connections;
    };

#ifndef NO_3DNN

    struct lineInfo {
        ulong IdA;
        ulong IdB;
        float R;
        Color C;
    };

    std::unordered_map<ulong, Vector3> NodeId2Pos;
    std::unordered_map<ulong, Color> NodId2Color;
    std::unordered_map<ulong, float> NodId2Size;
    std::unordered_map<ulong, Vector3> nodeId2RandPosDiff;
    std::vector<lineInfo> ConnectedNodesInfo;
    bool update3dLock = false;
    bool update3dCalcLock = false;
    bool canForceUnlock = false;
    NetworkGenome last3dNN;
    bool canClose3dNN = false;  // 用于中途关闭3d显示
    bool reRandomPosition = true;
    bool isOutputNodePosRand = false;

    bool isLast3dNN(NetworkGenome &NN) {
        if (NN.Neurons.size() != last3dNN.Neurons.size()) {
            mtx.lock();
            last3dNN = NN;
            mtx.unlock();
            return false;
        }

        for (ulong i = 0; i < NN.Neurons.size(); ++i) {
            if (NN.Neurons[i].Id != last3dNN.Neurons[i].Id) {
                mtx.lock();
                last3dNN = NN;
                mtx.unlock();
                return false;
            }
        }

        return true;
    };

    void Update3dNN(NetworkGenome NN, bool FL) {  // force unlock
        mtx.lock();
        if ((FL && canForceUnlock) || !update3dLock) {
            update3dLock = true;
            canForceUnlock = false;
            mtx.unlock();

            bool isLastNN = isLast3dNN(NN);
            if (!isLastNN || FL) {
                std::map<double, std::vector<ulong>> layer2Ids;

                for (auto &n: NN.Neurons) {
                    layer2Ids[n.Layer].push_back(n.Id);
                }

                float layerCount = 0;

                if (Opts.Enable3dRandPos && (!isLastNN || reRandomPosition)) {
                    for (auto &nn: NN.Neurons) {
                        if ((nn.Layer == 0.f && !isOutputNodePosRand) || (nn.Layer > 0.f && nn.Layer < 1.f)) {
                            nodeId2RandPosDiff[nn.Id] = {float(random() % 30) / 100.f - 0.15f, float(random() % 30) / 100.f - 0.15f, float(random() % 30) / 100.f - 0.15f};
                        }
                    }

                    if (!isOutputNodePosRand) {
                        isOutputNodePosRand = true;
                    }
                }

                std::unordered_map<ulong, Vector3> nodeId2Pos;

                for (auto &l2i: layer2Ids) {
                    if (l2i.first != 1.f) {
                        uint rows = uint(std::sqrt(float(l2i.second.size())));
                        uint columns = uint(l2i.second.size() / rows);
                        uint lastColumnsCount = l2i.second.size() % columns;
                        float startZ0 = -float(lastColumnsCount - 1) * Opts.Zy_Interval3d / 2.f;
                        float startZ1 = -float(columns - 1) * Opts.Zy_Interval3d / 2.f;
                        float startY = -float(rows - 1) * Opts.Zy_Interval3d / 2.f;
                        float thisZ = startZ1;
                        float thisY;
                        uint column = 0;

                        for (ulong i = 0; i < l2i.second.size(); ++i) {
                            if (!Opts.EnableCalc3dNN) {
                                if (l2i.first == 0.f) {
                                    NodId2Color[l2i.second[i]] = BLUE;
                                    NodId2Size[l2i.second[i]] = 0.1f;
                                } else {
                                    NodId2Color[l2i.second[i]] = YELLOW;
                                    float newNodeSize = NN.Neurons[l2i.second[i]].Bias / Opts.BiasRange * 0.3f;
                                    if (newNodeSize < 0.1f) {
                                        NodId2Size[l2i.second[i]] = 0.1f;
                                    } else {
                                        NodId2Size[l2i.second[i]] = newNodeSize;
                                    }
                                }
                            }

                            thisY = startY + Opts.Zy_Interval3d * float(column);

                            if (Opts.Enable3dRandPos) {
                                nodeId2Pos[l2i.second[i]] = {-(float(layer2Ids.size() - 1) * Opts.X_Interval3d / 2.f + nodeId2RandPosDiff[l2i.second[i]].x * Opts.X_Interval3d) + Opts.X_Interval3d * layerCount, -thisY + nodeId2RandPosDiff[l2i.second[i]].y * Opts.Zy_Interval3d,
                                                             thisZ + nodeId2RandPosDiff[l2i.second[i]].z * Opts.Zy_Interval3d};
                            } else {
                                nodeId2Pos[l2i.second[i]] = {-(float(layer2Ids.size() - 1) * Opts.X_Interval3d / 2.f) + Opts.X_Interval3d * layerCount, -thisY, thisZ};
                            }

                            thisZ += Opts.Zy_Interval3d;

                            if ((i + 1) % columns == 0) {
                                if (column + 1 == rows && lastColumnsCount > 0) {
                                    thisZ = startZ0;
                                } else {
                                    thisZ = startZ1;
                                }
                                ++column;
                            }
                        }
                    } else {
                        float startZ = -float(l2i.second.size() - 1) * Opts.Zy_Interval3d / 2.f;
                        for (ulong i = 0; i < l2i.second.size(); ++i) {
                            NodId2Color[l2i.second[i]] = RED;
                            NodId2Size[l2i.second[i]] = 0.1f;
                            float thisZ = startZ + Opts.Zy_Interval3d * float(i);
                            nodeId2Pos[l2i.second[i]] = {-(float(layer2Ids.size() - 1) * Opts.X_Interval3d / 2.f) + Opts.X_Interval3d * layerCount, 0.f, thisZ};
                        }
                    }
                    ++layerCount;
                }
                reRandomPosition = false;

                mtx.lock();
                NodeId2Pos = nodeId2Pos;
                mtx.unlock();
            }

            if (!Opts.EnableCalc3dNN) {
                std::vector<lineInfo> connectedNodesInfo;
                for (auto &conn: NN.Connections) {
                    if (conn.Enable) {
                        if (conn.Weight > 0) {
                            connectedNodesInfo.push_back(lineInfo{conn.ConnectedNeuronId[0], conn.ConnectedNeuronId[1], conn.Weight / Opts.WeightRange * 0.0045f + 0.0001f, ColorAlpha(RED, 0.5f)});
                        } else {
                            connectedNodesInfo.push_back(lineInfo{conn.ConnectedNeuronId[0], conn.ConnectedNeuronId[1], -conn.Weight / Opts.WeightRange * 0.0045f + 0.0001f, ColorAlpha(BLUE, 0.5f)});
                        }
                    }
                }
                mtx.lock();
                ConnectedNodesInfo = connectedNodesInfo;
                mtx.unlock();
            }

            mtx.lock();
            canForceUnlock = true;
            if (!FL) {
                mtx.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(Opts.Update3dIntercalMs));
                mtx.lock();
            }
            update3dLock = false;
            mtx.unlock();
        } else {
            mtx.unlock();
        }
    }

    void Update3dNN_Background(NetworkGenome NN, bool FL) {
        std::thread update3dnn(Update3dNN, NN, FL);
        update3dnn.detach();
    }

    void Show3dNN() {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);

        InitWindow(Opts.ScreenWidth, Opts.ScreenHeight, "SimpleNEAT NN");

        // Define the camera to look into our 3d world
        Camera3D camera = {0};
        camera.position = (Vector3) {0.f, 1.5f, 5.f}; // Camera position
        camera.target = (Vector3) {0.f, -.5f, 0.f};      // Camera looking at point
        camera.up = (Vector3) {0.f, 1.f, 0.f};          // Camera up vector (rotation towards target)
        camera.fovy = 60.0f;                                // Camera field-of-view Y
        camera.projection = CAMERA_PERSPECTIVE;                   // Camera mode type

        //        SetCameraMode(camera, CAMERA_FREE); // Set a free camera mode
        //        SetCameraAltControl(KEY_LEFT_SHIFT);
        //        SetCameraPanControl(MOUSE_BUTTON_LEFT);

        SetTargetFPS(60);                   // Set our game to run at 60 frames-per-second

        srandom((unsigned) clock());

        float setX_Interval3d = Opts.X_Interval3d;
        float setZy_Interval3d = Opts.Zy_Interval3d;

        while (!WindowShouldClose()) {     // Detect window close button or ESC key
            // Update
            UpdateCamera(&camera, CAMERA_ORBITAL);

            if (IsKeyPressed('Z')) {
                camera.target = (Vector3) {0.0f, 0.0f, 0.0f};
            }
            if (IsKeyPressed('R')) {
                Opts.X_Interval3d = setX_Interval3d;
                Opts.Zy_Interval3d = setZy_Interval3d;
                if (Opts.Enable3dRandPos) {
                    reRandomPosition = true;
                }
                Update3dNN_Background(last3dNN, true);
            }
            if (IsKeyDown('A')) {
                Opts.X_Interval3d -= .02f;
                Update3dNN_Background(last3dNN, true);
            }
            if (IsKeyDown('D')) {
                Opts.X_Interval3d += .02f;
                Update3dNN_Background(last3dNN, true);
            }
            if (IsKeyDown('W')) {
                Opts.Zy_Interval3d += .02f;
                Update3dNN_Background(last3dNN, true);
            }
            if (IsKeyDown('S')) {
                Opts.Zy_Interval3d -= .02f;
                Update3dNN_Background(last3dNN, true);
            }
            if (IsKeyPressed(KEY_SPACE)) {
                if (Opts.Enable3dRandPos) {
                    Opts.Enable3dRandPos = false;
                } else {
                    Opts.Enable3dRandPos = true;
                    reRandomPosition = true;
                }
                Update3dNN_Background(last3dNN, true);
            }

            // Draw
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

            auto nodeId2Pos = NodeId2Pos;
            auto connectedNodesInfo = ConnectedNodesInfo;

            for (auto &c: connectedNodesInfo) {
                //                DrawLine3D(nodeId2Pos[c.IdA], nodeId2Pos[c.IdB], c.C);
                DrawCylinderEx(nodeId2Pos[c.IdA], nodeId2Pos[c.IdB], c.R, c.R, 3, c.C);
            }

            for (auto &n: nodeId2Pos) {
                float cubeSize = NodId2Size[n.first];
                DrawCubeV(n.second, {cubeSize, cubeSize, cubeSize}, NodId2Color[n.first]);
            }

            EndMode3D();

            DrawFPS(10, 10);
            EndDrawing();

            if (canClose3dNN) {
                break;
            }
        }

        // De-Initialization
        CloseWindow();        // Close window and OpenGL context
    }

#endif

    class NeuralNetwork {
    public:
        std::map<std::array<ulong, 2>, ulong> HiddenNeuronInnovations;  // 只记录插入连接左右两个神经元id对应的隐藏层神经元id，新增神经元变异的时候全部个体需要检查唯一性，使用时必须使用mutex
        ulong FCHidenNeuronSize = 0;

        NetworkGenome NewNN();

        NetworkGenome NewFCNN();

        NetworkGenome SimplifyRemoveUselessConnectionRight(NetworkGenome nn);

        NetworkGenome SimplifyRemoveUselessConnectionLeft(NetworkGenome nn);

        NetworkGenome SimplifyRemoveDisable(NetworkGenome nn);

        std::vector<float> FeedForwardPredict(NetworkGenome *nn, std::vector<float> inputs, bool isAccelerate);

        std::vector<float> BackPropagation(NetworkGenome *nn, std::vector<float> inputs, std::vector<float> wants, bool isAccelerate);

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

        for (ulong i = 0; i < Opts.InputSize; ++i) {
            Neuron tmpNeuron = {.Id = i, .Bias = 0.f, .Layer = 0.,};
            newNeurons.push_back(tmpNeuron);
        }

        for (ulong i = 0; i < Opts.OutputSize; ++i) {
            ulong id = i + Opts.InputSize;
            Neuron tmpNeuron = {.Id = id, .Bias = float(random() % long(Opts.BiasRange * 200) - long(Opts.BiasRange * 100)) / 100, .Layer = 1.,};
            newNeurons.push_back(tmpNeuron);

            for (auto &n: newNeurons) {
                if (n.Layer == 0.) {
                    Connection tmpConnection = {.ConnectedNeuronId= {n.Id, id}, .Weight = float(random() % long(Opts.WeightRange * 200) - long(Opts.WeightRange * 100)) / 100, .Enable = true,};
                    newConnections.push_back(tmpConnection);
                    //                    if (ConnectionInnovations.find({n.Id, tmpNeuron.Id}) == ConnectionInnovations.end()) {
                    //                        ConnectionInnovations[{n.Id, tmpNeuron.Id}] = ConnectionInnovations.size();
                    //                    }
                }
            }
        }

        return NetworkGenome{.Neurons = newNeurons, .Connections = newConnections,};
    }

    NetworkGenome NeuralNetwork::NewFCNN() {  // 固定神经网络，输入隐藏层及对应神经元数量数
        if (Opts.InputSize <= 0 || Opts.OutputSize <= 0) {
            std::cerr << "Input or Output size fault: Input " << Opts.InputSize << ", Output " << Opts.OutputSize << std::endl;
            exit(0);
        }

        if (FCHidenNeuronSize == 0) {
            for (ulong &l: Opts.FCNN_hideLayers) {
                FCHidenNeuronSize += l;
            }
        }

        std::vector<Neuron> newNeurons;
        std::vector<Connection> newConnections;

        for (ulong i = 0; i < Opts.InputSize; ++i) {
            Neuron tmpNeuron = {.Id = i, .Bias = 1.f, .Layer = 0.,};
            newNeurons.push_back(tmpNeuron);
        }

        for (ulong i = 0; i < Opts.OutputSize; ++i) {
            ulong id = i + Opts.InputSize;
            Neuron tmpNeuron = {.Id = id, .Bias = float(random() % long(Opts.BiasRange * 200) - long(Opts.BiasRange * 100)) / 100, .Layer = 1.,};
            newNeurons.push_back(tmpNeuron);
        }

        double layerStep = 1. / double(Opts.FCNN_hideLayers.size() + 1);
        double thisLayer = 0.;
        ulong id = Opts.InputSize + Opts.OutputSize;

        for (ulong &l: Opts.FCNN_hideLayers) {
            thisLayer += layerStep;
            for (ulong i = 0; i < l; ++i) {
                Neuron tmpNeuron = {.Id = id, .Bias = float(random() % long(Opts.BiasRange * 200) - long(Opts.BiasRange * 100)) / 100, .Layer = thisLayer,};
                newNeurons.push_back(tmpNeuron);
                ++id;
            }
        }

        std::map<double, std::vector<Neuron *>> tmpLayerMap;  // 记录全部层，因为记录的是神经元地址，需要的时候才能临时生成记录
        double lastLayer = 0.;
        thisLayer = 0.;

        for (auto &n: newNeurons) {
            tmpLayerMap[n.Layer].push_back(&n);
        }

        for (auto &t: tmpLayerMap) {
            if (t.first != 0.) {
                for (auto &n: t.second) {
                    for (auto &ln: tmpLayerMap[lastLayer]) {
                        Connection tmpConnection = {.ConnectedNeuronId= {ln->Id, n->Id}, .Weight = float(random() % long(Opts.WeightRange * 200) - long(Opts.WeightRange * 100)) / 100, .Enable = true,};
                        newConnections.push_back(tmpConnection);
                    }
                }
            }
            lastLayer = thisLayer;
            thisLayer += layerStep;
        }

        return NetworkGenome{.Neurons = newNeurons, .Connections = newConnections,};
    }

    NetworkGenome NeuralNetwork::SimplifyRemoveUselessConnectionRight(NetworkGenome nn) { //合并中途凭空出现节点到右边的连接
        std::unordered_map<ulong, std::vector<Connection *>> remainingLeftIds;
        std::unordered_map<ulong, std::vector<Connection *>> remainingRightIds;
        std::unordered_map<ulong, std::vector<Connection *>> removeIds;
        std::unordered_map<ulong, Neuron> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址

        for (auto &n: nn.Neurons) {
            tmpNeuronMap[n.Id] = n;
        }

        for (;;) {
            remainingLeftIds.clear();
            remainingRightIds.clear();
            removeIds.clear();

            for (auto &c: nn.Connections) {
                remainingRightIds[c.ConnectedNeuronId[1]].push_back(&c);   // 添加所有连接右边的神经元id
            }

            for (auto &c: nn.Connections) {
                if (remainingRightIds.find(c.ConnectedNeuronId[0]) != remainingRightIds.end() || tmpNeuronMap[c.ConnectedNeuronId[0]].Layer == 0.) {  // c++17
                    //if (remainingRightIds.contains(n.ConnectedNeuronId[0]) || tmpNeuronMap[n.ConnectedNeuronId[0]].Layer == 0.f) {   // c++20
                    remainingLeftIds[c.ConnectedNeuronId[0]].push_back(&c);  // 添加所左边的神经元id（可作为另外一条连接的右边神经元）(左边还有连接)
                } else {
                    removeIds[c.ConnectedNeuronId[0]].push_back(&c);
                }
            }

            for (auto &r: removeIds) {
                float thisBias = tmpNeuronMap[r.first].Bias;
                for (auto &cp: r.second) {
                    tmpNeuronMap[cp->ConnectedNeuronId[1]].Bias += Opts.ActiveFunction(thisBias) * cp->Weight;
                }
            }

            std::vector<Neuron> newNeurons;
            std::vector<Connection> newConnections;

            for (auto &r: remainingLeftIds) {
                newNeurons.push_back(tmpNeuronMap[r.first]);
                for (auto &cp: r.second) {
                    newConnections.push_back(*cp);
                }
            }

            nn = {.Neurons=newNeurons, .Connections=newConnections,};

            if (removeIds.empty()) {
                break;
            }
        }

        for (auto &n: tmpNeuronMap) {
            if (n.second.Layer == 1.) {
                nn.Neurons.push_back(n.second);
            }
        }

        return nn;
    }

    NetworkGenome NeuralNetwork::SimplifyRemoveUselessConnectionLeft(NetworkGenome nn) { // 实现从左到右的无效连接移除
        std::unordered_map<ulong, std::vector<Connection *>> remainingLeftIds;
        std::unordered_map<ulong, std::vector<Connection *>> remainingRightIds;
        std::unordered_map<ulong, Neuron> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址

        for (auto &n: nn.Neurons) {
            tmpNeuronMap[n.Id] = n;
        }

        for (;;) {
            bool isFinished = true;
            remainingLeftIds.clear();
            remainingRightIds.clear();

            for (auto &c: nn.Connections) {
                remainingLeftIds[c.ConnectedNeuronId[0]].push_back(&c);   // 添加所有连接左边的神经元id
            }


            for (auto &c: nn.Connections) {
                if (remainingLeftIds.find(c.ConnectedNeuronId[1]) != remainingLeftIds.end() || tmpNeuronMap[c.ConnectedNeuronId[1]].Layer == 1.) {
                    //                if (remainingLeftIds.contains(n.ConnectedNeuronId[1]) || tmpNeuronMap[n.ConnectedNeuronId[1]].Layer == 1.f) {
                    remainingRightIds[c.ConnectedNeuronId[1]].push_back(&c);  // 添加所右边的神经元id（可作为另外一条连接的左边神经元）(右边还有连接)
                } else {
                    isFinished = false;
                }
            }

            std::vector<Neuron> newNeurons;
            std::vector<Connection> newConnections;

            for (auto &r: remainingRightIds) {
                newNeurons.push_back(tmpNeuronMap[r.first]);
                for (auto &cp: r.second) {
                    newConnections.push_back(*cp);
                }
            }

            nn = {.Neurons=newNeurons, .Connections=newConnections,};

            if (isFinished) {
                break;
            }
        }

        for (auto &n: tmpNeuronMap) {
            if (n.second.Layer == 0.) {
                nn.Neurons.push_back(n.second);
            }
        }

        return nn;
    }

    NetworkGenome NeuralNetwork::SimplifyRemoveDisable(NetworkGenome nn) {
        std::vector<Connection> newConnections;
        std::unordered_map<ulong, uint> remainingIds;

        for (auto c: nn.Connections) {
            if (c.Enable && std::abs(c.Weight) > 0.001f) {  // Weight绝对值小于0.001算成disable
                newConnections.push_back(c);
                remainingIds[c.ConnectedNeuronId[0]] = 0;
                remainingIds[c.ConnectedNeuronId[1]] = 0;
            }
        }

        std::unordered_map<ulong, Neuron *> tmpNeuronMap;

        for (auto &n: nn.Neurons) {
            tmpNeuronMap[n.Id] = &n;
            if (n.Layer == 0. || n.Layer == 1.) {
                remainingIds[n.Id] = 0;
            }
        }

        std::vector<Neuron> newNeurons;

        for (auto &i: remainingIds) {
            newNeurons.push_back(*tmpNeuronMap[i.first]);
        }

        return NetworkGenome{.Neurons = newNeurons, .Connections = newConnections,};
    }

    std::vector<float> NeuralNetwork::FeedForwardPredict(NetworkGenome *nn, std::vector<float> inputs, bool isAccelerate) {  // 在使用CPU多核运算时，由于多线程开销问题，神经网络结构太简单反而会运算得更慢
        std::unordered_map<ulong, Neuron *> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址
        std::map<double, std::vector<Neuron *>> tmpLayerMap;  // 记录层对应神经元，同上因为记录的是神经元地址，需要的时候才能临时生成记录

        for (auto &n: nn->Neurons) {
            tmpNeuronMap[n.Id] = &n;
            tmpLayerMap[n.Layer].push_back(&n);
        }

        if (tmpLayerMap[0.].size() != Opts.InputSize || tmpLayerMap[1.].size() != Opts.OutputSize) {
            std::cerr << "NeuralNetwork Nodes Error: Opts.InputSize " << Opts.InputSize << " Input Layer Size: " << tmpLayerMap[0.].size() << " Opts.OutputSize: " << Opts.OutputSize << " Output Layer Size: " << tmpLayerMap[1.].size() << std::endl;
            std::exit(0);
        }

        if (Opts.InputSize != inputs.size()) {
            std::cerr << "FeedForwardPredict: Input length " << inputs.size() << " diffrent with NN input nodes " << Opts.InputSize << std::endl;
            std::exit(0);
        }

        std::unordered_map<ulong, float> tmpNodesOutput;

        std::function<void(ulong)> calculateNeuron = [&](ulong nid) {
            float thisOutput = 0.f;

            for (auto &connection: nn->Connections) {
                if (connection.ConnectedNeuronId[1] == nid && connection.Enable) {
                    if (isAccelerate) {
                        std::shared_lock<std::shared_mutex> lock(mtx);
                        thisOutput += tmpNodesOutput[connection.ConnectedNeuronId[0]] * connection.Weight;
                    } else {
                        thisOutput += tmpNodesOutput[connection.ConnectedNeuronId[0]] * connection.Weight;
                    }
                }
            }

            thisOutput = Opts.ActiveFunction(thisOutput + tmpNeuronMap[nid]->Bias);

            mtx.lock();
            tmpNodesOutput[nid] = thisOutput;
            mtx.unlock();
        };

        uint i = 0;
        for (auto &n: tmpLayerMap[0.]) { // 初始化输入节点
            tmpNodesOutput[n->Id] = inputs[i];
            ++i;
        }

        std::vector<float> outputs(Opts.OutputSize);

        for (auto &l: tmpLayerMap) {    // 神经元根据layer排序
            if (l.first == 0.) {   // 跳过输入节点
                continue;
            }

            if (isAccelerate) {
                for (auto &n: l.second) {
                    tPool.push_task(calculateNeuron, n->Id);
                }

                tPool.wait_for_tasks();

                if (l.first == 1.) {  // 输出神经元
                    int outputCount = 0;
                    for (auto &n: l.second) {
                        outputs[outputCount] = tmpNodesOutput[n->Id];
                        ++outputCount;
                    }
                }
            } else {
                int outputCount = 0;
                for (auto &n: l.second) {
                    calculateNeuron(n->Id);  // 计算隐藏和输出神经元
                    if (l.first == 1.) {  // 输出神经元
                        outputs[outputCount] = tmpNodesOutput[n->Id];
                        ++outputCount;
                    }
                }
            }
        }

#ifndef NO_3DNN

        mtx.lock();
        if (Opts.Enable3dNN && Opts.EnableCalc3dNN && !update3dCalcLock) {
            update3dCalcLock = true;
            mtx.unlock();

            std::unordered_map<ulong, Color> nodId2Color;
            std::unordered_map<ulong, float> nodId2Size;

            for (auto &n: tmpNodesOutput) {
                if (n.second > 0.1f) {
                    if (tmpNeuronMap[n.first]->Layer == 0.f || tmpNeuronMap[n.first]->Layer == 1.f) {
                        nodId2Color[n.first] = WHITE;
                    } else {
                        nodId2Color[n.first] = GRAY;
                    }
                    nodId2Size[n.first] = 0.1f * (n.second + 0.9f);
                } else {
                    if (tmpNeuronMap[n.first]->Layer == 0.f) {
                        nodId2Color[n.first] = BLUE;
                    } else if (tmpNeuronMap[n.first]->Layer == 1.f) {
                        nodId2Color[n.first] = RED;
                    } else {
                        nodId2Color[n.first] = YELLOW;
                    }
                    nodId2Size[n.first] = 0.1f;
                }
            }

            std::vector<lineInfo> connectedNodesInfo;

            if (Opts.EnableCalc3dNN) {
                for (auto &c: nn->Connections) {
                    if (nodId2Size[c.ConnectedNeuronId[0]] > 0.1f && nodId2Size[c.ConnectedNeuronId[1]] > 0.1f && c.Enable && c.Weight > 0.f) {
                        connectedNodesInfo.push_back(lineInfo{c.ConnectedNeuronId[0], c.ConnectedNeuronId[1], c.Weight / Opts.WeightRange * 0.009f + 0.0001f, ColorAlpha(WHITE, 0.3f)});
                    }
                }
            }

            mtx.lock();
            NodId2Color = nodId2Color;
            NodId2Size = nodId2Size;
            ConnectedNodesInfo = connectedNodesInfo;
            mtx.unlock();

            Update3dNN_Background(*nn, false);

            std::thread updateNNCalc([]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(Opts.Update3dIntercalMs));

                mtx.lock();
                update3dCalcLock = false;
                mtx.unlock();
            });
            updateNNCalc.detach();
        } else {
            mtx.unlock();
        }

#endif

        return outputs;
    };

    std::vector<float> NeuralNetwork::BackPropagation(NetworkGenome *nn, std::vector<float> inputs, std::vector<float> wants, bool isAccelerate) {  // 如果当前预测fitness大于预设，则判断为解决问题，返回计算结果, 在使用CPU多核运算时，由于多线程开销问题，神经网络结构太简单反而会运算得更慢 TODO: 权重和偏置范围该怎么限制?丢弃?
        std::unordered_map<ulong, Neuron *> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址
        std::map<double, std::vector<Neuron *>> tmpLayerMap;  // 记录层对应神经元，同上因为记录的是神经元地址，需要的时候才能临时生成记录

        for (auto &n: nn->Neurons) {
            tmpNeuronMap[n.Id] = &n;
            tmpLayerMap[n.Layer].push_back(&n);
        }


        if (Opts.InputSize != inputs.size()) {
            std::cerr << "BackPropagation: Input length " << inputs.size() << " diffrent with NN input nodes " << Opts.InputSize << std::endl;
            std::exit(0);
        }

        std::unordered_map<ulong, float> tmpNodesOutput;

        std::function<void(ulong)> calculateNeuron = [&](ulong nid) {
            float thisOutput = 0.f;

            for (auto &connection: nn->Connections) {
                if (connection.ConnectedNeuronId[1] == nid && connection.Enable) {
                    if (isAccelerate) {
                        std::shared_lock<std::shared_mutex> lock(mtx);
                        thisOutput += tmpNodesOutput[connection.ConnectedNeuronId[0]] * connection.Weight;
                    } else {
                        thisOutput += tmpNodesOutput[connection.ConnectedNeuronId[0]] * connection.Weight;
                    }
                }
            }

            float tmpNeuronBias = tmpNeuronMap[nid]->Bias;

            thisOutput = Opts.ActiveFunction(thisOutput + tmpNeuronBias);

            mtx.lock();
            tmpNodesOutput[nid] = thisOutput;
            mtx.unlock();
        };

        uint i = 0;
        for (auto &n: tmpLayerMap[0.]) { // 初始化输入节点
            tmpNodesOutput[n->Id] = inputs[i];
            ++i;
        }

        std::vector<float> outputs(Opts.OutputSize);

        for (auto &l: tmpLayerMap) {    // 神经元根据layer排序
            if (l.first == 0.) {   // 跳过输入节点
                continue;
            }

            if (isAccelerate) {
                for (auto &n: l.second) {
                    tPool.push_task(calculateNeuron, n->Id);
                }

                tPool.wait_for_tasks();

                if (l.first == 1.) {  // 输出神经元
                    int outputCount = 0;
                    for (auto &n: l.second) {
                        outputs[outputCount] = tmpNodesOutput[n->Id];
                        ++outputCount;
                    }
                }
            } else {
                int outputCount = 0;
                for (auto &n: l.second) {
                    calculateNeuron(n->Id);  // 计算隐藏和输出神经元
                    if (l.first == 1.) {  // 输出神经元
                        outputs[outputCount] = tmpNodesOutput[n->Id];
                        ++outputCount;
                    }
                }
            }
        }

        // 上面是正向计算全部节点输出,接下来开始反向传播
        // 先计算每个节点的误差,偏导数又称为误差项也称为灵敏度

        std::unordered_map<ulong, float> tmpNodesOutputError;

        std::function<void(ulong)> calculateNeuronError = [&](ulong nid) {
            float thisOutputError = 0.f;

            for (auto &connection: nn->Connections) {
                if (connection.ConnectedNeuronId[0] == nid && connection.Enable) {
                    if (isAccelerate) {
                        std::shared_lock<std::shared_mutex> lock(mtx);
                        thisOutputError += tmpNodesOutputError[connection.ConnectedNeuronId[1]] * connection.Weight;
                    } else {
                        thisOutputError += tmpNodesOutputError[connection.ConnectedNeuronId[1]] * connection.Weight;
                    }
                }
            }

            float tmpNodesOutputNid = tmpNodesOutput[nid];

            thisOutputError *= Opts.DerivativeFunction(tmpNodesOutputNid) * tmpNodesOutputNid;

            mtx.lock();
            tmpNodesOutputError[nid] = thisOutputError;
            mtx.unlock();
        };

        uint wantsCount = 0;
        for (std::map<double, std::vector<Neuron *>>::reverse_iterator ri = tmpLayerMap.rbegin(); ri != tmpLayerMap.rend(); ++ri) {
            for (auto &n: ri->second) {
                if (ri->first == 1.f) {   // 计算输出神经元点误差
                    tmpNodesOutputError[n->Id] = Opts.DerivativeFunction(tmpNodesOutput[n->Id]) * (wants[wantsCount] - tmpNodesOutput[n->Id]);
                    ++wantsCount;
                    continue;
                }

                if (ri->first == 0.f) {
                    break;
                }

                if (isAccelerate) {
                    tPool.push_task(calculateNeuronError, n->Id);
                } else {
                    calculateNeuronError(n->Id);  // 计算隐藏神经元误差
                }
            }

            if (isAccelerate) {
                tPool.wait_for_tasks();
            }
        }


        for (auto &connection: nn->Connections) { // 更新连接权重
            float newWeight = connection.Weight + Opts.LearnRate * tmpNodesOutputError[connection.ConnectedNeuronId[1]] * tmpNodesOutput[connection.ConnectedNeuronId[0]];
            if (std::abs(newWeight) > Opts.WeightRange) {
//                std::cerr << "Weight out of range: [" << connection.ConnectedNeuronId[0] << "," << connection.ConnectedNeuronId[1] << "], " << connection.Weight << " -> " << newWeight << "\n";
            } else {
                connection.Weight = newWeight;
            }

        }

        for (auto &n: nn->Neurons) { // 更新神经元偏置
            if (n.Layer != 0.) {
                float newBias = n.Bias + Opts.LearnRate * tmpNodesOutputError[n.Id];
                if (std::abs(newBias) > Opts.BiasRange) {
//                    std::cerr << "Bias out of range: [" << n.Id << "], " << n.Bias << " -> " << newBias << "\n";
                } else {
                    n.Bias = newBias;
                }
            }
        }

        //        nn->Age = 0;

        if (outputs.size() != wants.size()) {
            std::cerr << "BackPropagation: outputs.size(" << outputs.size() << ") != wants.size(" << wants.size() << ")\n";
            exit(0);
        }

        return outputs;
    };

    void NeuralNetwork::ExportNNToDot(NetworkGenome &nn, std::string fileName) {
        std::string data = "digraph G {\n    rankdir=\"LR\";\n";
        ulong inId = 0;
        ulong outId = 0;

        for (auto &n: nn.Neurons) {
            std::string line;
            std::stringstream streamBias;
            if (n.Layer == 0.) {
                line = "    subgraph cluster0{" + std::to_string(n.Id) + " [fontsize=24,width=0,height=0,color=lightblue,style=filled,shape=component,width=1,height=1,label=\"Input_" + std::to_string(inId) + "\"]}\n";
                ++inId;
            } else if (n.Layer == 1.) {
                streamBias << std::setprecision(3) << n.Bias;
                line = "    subgraph cluster1{" + std::to_string(n.Id) + " [fontsize=24,width=0,height=0,color=lightgray,style=filled,shape=diamond,width=1,height=1,label=\"Output_" + std::to_string(outId) + "\\n(" + streamBias.str() + ")\"]}\n";
                ++outId;
            } else {
                streamBias << std::setprecision(3) << n.Bias;
                //                line = "    subgraph cluster" + std::to_string(int(n.Layer * float(1000000000))) + "{" + std::to_string(n.Id) + " [fontsize=27,width=0,height=0,color=lightgreen,style=filled,shape=box,width=1,height=1,label=\"(" + streamBias.str() + ")\"]}\n";
                line = "    " + std::to_string(n.Id) + " [fontsize=27,width=0,height=0,color=lightgreen,style=filled,shape=box,width=1,height=1,label=\"(" + streamBias.str() + ")\"]\n";
            }
            data += line;
        }

        for (auto &c: nn.Connections) {
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

        for (auto &n: nn.Neurons) {
            std::stringstream streamLayer;
            streamLayer << std::scientific << n.Layer;
            std::stringstream streamBias;
            streamBias << std::scientific << n.Bias;
            data += std::to_string(n.Id) + "," + streamLayer.str() + "," + streamBias.str() + "\n";
        }

        data += "~\n";

        for (auto &c: nn.Connections) {
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

        ulong InputSize = 0;
        ulong OutputSize = 0;

        std::vector<Neuron> newNeurons;
        std::vector<Connection> newConnections;
        std::unordered_map<ulong, ulong> tmpIdMap;

        int dataType = 0;
        while (getline(input_file, line)) {
            if (line == "~") {
                ++dataType;
                continue;
            }

            if (dataType == 0) {
                auto datas = SplitString(line, ",");
                InputSize = std::stoul(datas[0]);
                OutputSize = std::stoul(datas[1]);
                continue;
            }

            if (dataType == 1) {
                auto datas = SplitString(line, ",");
                double nnLayer = std::stod(datas[1]);
                ulong id = std::stoul(datas[0]);
                newNeurons.push_back(Neuron{.Id = id, .Bias = std::stof(datas[2]), .Layer = nnLayer,});
                tmpIdMap[id] = 0;
                continue;
            }

            auto datas = SplitString(line, ",");
            auto cnid = SplitString(datas[0], "->");
            newConnections.push_back(Connection{.ConnectedNeuronId = {std::stoul(cnid[0]), std::stoul(cnid[1])}, .Weight = std::stof(datas[1]), .Enable = true,});
        }

        input_file.close();

        if (HiddenNeuronInnovations.empty() && (Opts.InputSize > InputSize || Opts.OutputSize > OutputSize)) {
            std::cerr << "HiddenNeuronInnovations is empty, import .innov file fisrt.";
            exit(0);
        }

        ulong inInnovMapSize = 0;
        for (auto &n: HiddenNeuronInnovations) {
            //            if (tmpIdMap.contains(n.second)) {
            if (tmpIdMap.find(n.second) != tmpIdMap.end()) {
                ++inInnovMapSize;
            }
        }

        FCHidenNeuronSize = newNeurons.size() - InputSize - OutputSize - inInnovMapSize;

        if (Opts.InputSize > InputSize) {
            for (ulong i = 0; i < Opts.InputSize - InputSize; ++i) {
                newNeurons.push_back(Neuron{.Id = HiddenNeuronInnovations.size() + OutputSize + InputSize + i + FCHidenNeuronSize, .Bias = 0.f, .Layer = 0.,});
            }
        }

        if (Opts.OutputSize > OutputSize) {
            for (ulong i = 0; i < Opts.OutputSize - OutputSize; ++i) {
                newNeurons.push_back(Neuron{.Id = HiddenNeuronInnovations.size() + OutputSize + i + Opts.InputSize + FCHidenNeuronSize, .Bias = float(random() % long(Opts.BiasRange * 200) - long(Opts.BiasRange * 100)) / 100, .Layer = 1.,});
            }
        }

        return NetworkGenome{.Neurons = newNeurons, .Connections = newConnections,};
    }

    void NeuralNetwork::ExportInnovations(std::string fileName) {
        std::string data;

        for (auto &i: HiddenNeuronInnovations) {
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
            HiddenNeuronInnovations[{std::stoul(datas[0]), std::stoul(datas[1])}] = std::stoul(datas[2]);
        }
    }

}

#endif //MYNEAT_NERAULNETWORK_HPP

