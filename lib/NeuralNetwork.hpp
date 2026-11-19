#pragma once

#ifndef MYNEAT_NERAULNETWORK_HPP
#define MYNEAT_NERAULNETWORK_HPP

#include <vector>
#include <map>
#include <array>
#include <functional>
#include <fstream>
#include <mutex>
#include "Option.hpp"

#ifndef NO_3DNN

#include "raylib.h"

#endif

namespace znn {
    struct Neuron {  // 定义神经元结构体
        ulong Id;  // 神经元ID
        float Bias;  // 神经元偏置
        double Layer;  // 神经元的逻辑位置（层）
        float momentum = 0.f;  // [OPT] BP偏置更新动量速度（不导出；ImportNN 创建时默认0，动量语义见 Opts.BPMomentum）
    };

    struct Connection {  // 定义神经连接结构体
        std::array<ulong, 2> ConnectedNeuronId;  // 神经连接由两个神经元而定
        float Weight; // 连接的权重
        bool Enable; // 连接是否启用
        float momentum = 0.f;  // [OPT] BP权重更新动量速度（不导出；ImportNN 创建时默认0）
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
    bool isInputNodePosRand = false;

    bool isLast3dNN(NetworkGenome &NN) {
        // [FIX] 原实现先在锁外比较 size/Id 再在锁内赋值，比较本身处于竞态窗口（另一线程可能正在写
        // last3dNN）；把比较和赋值统一放到锁内
        mtx.lock();
        bool isSame = true;
        if (NN.Neurons.size() != last3dNN.Neurons.size()) {
            isSame = false;
        } else {
            for (ulong i = 0; i < NN.Neurons.size(); ++i) {
                if (NN.Neurons[i].Id != last3dNN.Neurons[i].Id) {
                    isSame = false;
                    break;
                }
            }
        }
        if (!isSame) {
            last3dNN = NN;
        }
        mtx.unlock();
        return isSame;
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
                        if ((nn.Layer == 0.f && !isInputNodePosRand) || (nn.Layer > 0.f && nn.Layer < 1.f)) {
                            nodeId2RandPosDiff[nn.Id] = {float(random() % 30) / 100.f - 0.15f, float(random() % 30) / 100.f - 0.15f, float(random() % 30) / 100.f - 0.15f};
                        }
                    }

                    if (!isInputNodePosRand) {
                        isInputNodePosRand = true;
                    }
                }

                std::unordered_map<ulong, Vector3> nodeId2Pos;
                // [FIX] 原代码直接无锁写全局 NodId2Color/NodId2Size，渲染线程同时无锁读（unordered_map 并发读写 UB）；
                // 改为先写局部 map，最后与 NodeId2Pos 一起在锁内整体替换
                std::unordered_map<ulong, Color> nodId2Color;
                std::unordered_map<ulong, float> nodId2Size;
                // [FIX] 原代码把神经元 Id 当 Neurons 数组下标用（NN.Neurons[id]），交叉/简化后的基因组
                // Id 与下标不对应，会读错或越界；改为按 Id 查表
                std::unordered_map<ulong, Neuron *> tmpNeuronMap;
                for (auto &n: NN.Neurons) {
                    tmpNeuronMap[n.Id] = &n;
                }

                for (auto &l2i: layer2Ids) {
                    if (l2i.first != 1.f || !Opts.Enable3dOutputOneLine) {
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
                            if (!Opts.Enable3dCalc) {
                                if (l2i.first == 0.f) {
                                    nodId2Color[l2i.second[i]] = BLUE;
                                    nodId2Size[l2i.second[i]] = 0.1f;
                                } else {
                                    nodId2Color[l2i.second[i]] = YELLOW;
                                    // [FIX] 原代码 NN.Neurons[l2i.second[i]] 把 Id 当下标；改为按 Id 查表
                                    float newNodeSize = tmpNeuronMap[l2i.second[i]]->Bias / Opts.BiasRange * 0.3f;
                                    if (newNodeSize < 0.1f) {
                                        nodId2Size[l2i.second[i]] = 0.1f;
                                    } else {
                                        nodId2Size[l2i.second[i]] = newNodeSize;
                                    }
                                }
                            }

                            thisY = startY + Opts.Zy_Interval3d * float(column);

                            if (Opts.Enable3dRandPos && l2i.first != 1.f) {
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
                            nodId2Color[l2i.second[i]] = RED;
                            nodId2Size[l2i.second[i]] = 0.1f;
                            float thisZ = startZ + Opts.Zy_Interval3d * float(i);
                            nodeId2Pos[l2i.second[i]] = {-(float(layer2Ids.size() - 1) * Opts.X_Interval3d / 2.f) + Opts.X_Interval3d * layerCount, 0.f, thisZ};
                        }
                    }
                    ++layerCount;
                }
                reRandomPosition = false;

                // [FIX] 三个渲染用的全局 map 统一在锁内一次性替换，消除渲染线程的半更新/并发读写窗口
                mtx.lock();
                NodeId2Pos = nodeId2Pos;
                NodId2Color = nodId2Color;
                NodId2Size = nodId2Size;
                mtx.unlock();
            }

            if (!Opts.Enable3dCalc) {
                std::vector<lineInfo> connectedNodesInfo;
                for (auto &conn: NN.Connections) {
                    if (conn.Enable) {
                        if (conn.Weight > 0) {
                            connectedNodesInfo.push_back(lineInfo{conn.ConnectedNeuronId[0], conn.ConnectedNeuronId[1], conn.Weight / Opts.WeightRange * 0.03f + 0.0001f, ColorAlpha(RED, 0.5f)});
                        } else {
                            connectedNodesInfo.push_back(lineInfo{conn.ConnectedNeuronId[0], conn.ConnectedNeuronId[1], -conn.Weight / Opts.WeightRange * 0.03f + 0.0001f, ColorAlpha(BLUE, 0.5f)});
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

        // [FIX] 原代码 srandom(clock()) 在渲染线程重置全局 RNG，会破坏主线程的随机序列（训练不可复现）
        // srandom((unsigned) clock());

        float setX_Interval3d = Opts.X_Interval3d;
        float setZy_Interval3d = Opts.Zy_Interval3d;

        while (!WindowShouldClose()) {     // Detect window close button or ESC key
            // Update
            UpdateCamera(&camera, CAMERA_ORBITAL);

            // [FIX] A/S/W/D 原为 IsKeyDown：按住时每帧 spawn 一个 detached 线程，数量无上界；
            // 改为 IsKeyPressed，按键一次只触发一次更新
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
            if (IsKeyPressed('A')) {  // [FIX] IsKeyDown -> IsKeyPressed：按住不再每帧 spawn 线程
                Opts.X_Interval3d -= .02f;
                Update3dNN_Background(last3dNN, true);
            }
            if (IsKeyPressed('D')) {  // [FIX] 同上
                Opts.X_Interval3d += .02f;
                Update3dNN_Background(last3dNN, true);
            }
            if (IsKeyPressed('W')) {  // [FIX] 同上
                Opts.Zy_Interval3d += .02f;
                Update3dNN_Background(last3dNN, true);
            }
            if (IsKeyPressed('S')) {  // [FIX] 同上
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

            // [FIX] 原代码直接拷贝全局 NodeId2Pos/ConnectedNodesInfo/NodId2Size/NodId2Color，
            // 与后台更新线程并发读写 unordered_map 是 UB（偶发崩溃来源）；
            // 改为锁内一次性快照，同一帧内数据自洽
            mtx.lock();
            auto nodeId2Pos = NodeId2Pos;
            auto connectedNodesInfo = ConnectedNodesInfo;
            auto nodId2Size = NodId2Size;
            auto nodId2Color = NodId2Color;
            mtx.unlock();

            BeginMode3D(camera);

            //                DrawCubeV({1.f, 0.f, 0.f}, {0.03f, 0.03f, 0.03f}, RED);
            //                DrawCubeV({0.f, 1.f, 0.f}, {0.03f, 0.03f, 0.03f}, YELLOW);
            //                DrawCubeV({0.f, 0.f, 1.f}, {0.03f, 0.03f, 0.03f}, BLUE);
            //
            //                DrawLine3D({-1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, RED);
            //                DrawLine3D({0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}, YELLOW);
            //                DrawLine3D({0.f, 0.f, -1.f}, {0.f, 0.f, 1.f}, BLUE);

            for (auto &c: connectedNodesInfo) {
                //                DrawLine3D(nodeId2Pos[c.IdA], nodeId2Pos[c.IdB], c.C);
                DrawCylinderEx(nodeId2Pos[c.IdA], nodeId2Pos[c.IdB], c.R, c.R, 3, c.C);
            }

            for (auto &n: nodeId2Pos) {
                // [FIX] NodId2Size/NodId2Color 已在上面锁内快照为局部副本，这里改用快照，消除无锁读
                float cubeSize = nodId2Size[n.first];
                DrawCubeV(n.second, {cubeSize, cubeSize, cubeSize}, nodId2Color[n.first]);
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
        std::map<std::array<ulong, 2>, ulong> HiddenNeuronInnovations;  // 记录插入连接左右两个神经元id对应的隐藏层神经元id，新增神经元变异的时候全部个体需要检查唯一性，使用时必须使用mutex
        ulong FCHidenNeuronSize = 0;

        NetworkGenome NewNN();

        NetworkGenome NewFCNN();

        NetworkGenome SimplifyRemoveUselessConnectionRight(NetworkGenome nn);

        NetworkGenome SimplifyRemoveUselessConnectionLeft(NetworkGenome nn);

        NetworkGenome SimplifyRemoveDisable(NetworkGenome nn);

        // [OPT] inputs/wants 改 const 引用：每样本×每个体×每代调用，按值传参整份拷贝是纯开销
        std::vector<float> FeedForwardPredict(NetworkGenome *nn, const std::vector<float> &inputs, bool isAccelerate);

        std::vector<float> FCNNFeedForwardPredict(NetworkGenome *fh, const std::vector<float> &inputs, bool isAccelerate);

        std::vector<float> BackPropagation(NetworkGenome *nn, const std::vector<float> &inputs, const std::vector<float> &wants, bool isAccelerate);

        std::vector<float> FCNNBackPropagation(NetworkGenome *nn, const std::vector<float> &inputs, const std::vector<float> &wants, bool isAccelerate);

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
            Neuron tmpNeuron = {.Id = id, .Bias = float(random() % long(Opts.NewNNBiasRange * 200) - long(Opts.NewNNBiasRange * 100)) / 100.f, .Layer = 1.,};
            newNeurons.push_back(tmpNeuron);

            for (auto &n: newNeurons) {
                if (n.Layer == 0.) {
                    Connection tmpConnection = {.ConnectedNeuronId= {n.Id, id}, .Weight = float(random() % long(Opts.NewNNWeightRange * 200) - long(Opts.NewNNWeightRange * 100)) / 100.f, .Enable = true,};
                    newConnections.push_back(tmpConnection);
                    //                    if (ConnectionInnovations.find({n.Id, tmpNeuron.Id}) == ConnectionInnovations.end()) {
                    //                        ConnectionInnovations[{n.Id, tmpNeuron.Id}] = ConnectionInnovations.size();
                    //                    }
                }
            }
        }

        return NetworkGenome{.Neurons = newNeurons, .Connections = newConnections,};
    }

    NetworkGenome NeuralNetwork::NewFCNN() {  // 跨层全连接神经网络，输入隐藏层及对应神经元数量数
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
            Neuron tmpNeuron = {.Id = id, .Bias = float(random() % long(Opts.NewNNBiasRange * 200) - long(Opts.NewNNBiasRange * 100)) / 100.f, .Layer = 1.,};
            newNeurons.push_back(tmpNeuron);
        }

        double layerStep = 1. / double(Opts.FCNN_hideLayers.size() + 1);
        double thisLayer = 0.;
        ulong id = Opts.InputSize + Opts.OutputSize;

        for (ulong &l: Opts.FCNN_hideLayers) {
            thisLayer += layerStep;
            for (ulong i = 0; i < l; ++i) {
                Neuron tmpNeuron = {.Id = id, .Bias = float(random() % long(Opts.NewNNBiasRange * 200) - long(Opts.NewNNBiasRange * 100)) / 100.f, .Layer = thisLayer,};
                newNeurons.push_back(tmpNeuron);
                ++id;
            }
        }

        std::map<double, std::vector<Neuron *>> tmpLayerMap;  // 记录全部层，因为记录的是神经元地址，需要的时候才能临时生成记录

        for (auto &n: newNeurons) {
            tmpLayerMap[n.Layer].push_back(&n);
        }

        double lastLayer = 0.;
        thisLayer = 0.;
        std::vector<double> allLayerId;
        std::map<double, std::vector<Neuron *>>::iterator it = tmpLayerMap.begin();

        for (auto &t: tmpLayerMap) {
            if (t.first != 0.) {
                for (auto &n: t.second) {
                    if (Opts.FFCNNInsteadOfFCNN) {
                        for (auto &tt: tmpLayerMap) {
                            for (auto &ln: tt.second) {
                                Connection tmpConnection = {.ConnectedNeuronId= {ln->Id, n->Id}, .Weight = float(random() % long(Opts.NewNNWeightRange * 200) - long(Opts.NewNNWeightRange * 100)) / 100.f, .Enable = true,};
                                newConnections.push_back(tmpConnection);
                            }
                            if (tt.first == lastLayer) {
                                break;
                            }
                        }
                    } else {
                        for (auto &ln: tmpLayerMap[lastLayer]) {
                            Connection tmpConnection = {.ConnectedNeuronId= {ln->Id, n->Id}, .Weight = float(random() % long(Opts.NewNNWeightRange * 200) - long(Opts.NewNNWeightRange * 100)) / 100.f, .Enable = true,};
                            newConnections.push_back(tmpConnection);
                        }
                    }
                }
            }

            if (t.first < 1.) {
                lastLayer = thisLayer;

                it = std::next(it);
                thisLayer = it->first;
            }
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

            // [FIX] 无任何输出连接的输入神经元原本会被丢掉，压缩后 Neurons 中输入节点数量与
            // Opts.InputSize 不一致，FeedForwardPredict 会报错退出；补录全部 layer 0 神经元
            for (auto &n: tmpNeuronMap) {
                if (n.second.Layer == 0.) {
                    remainingLeftIds[n.first];  // operator[] 插入空向量：只保留节点本身，不加连接
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

#ifndef NO_3DNN

    void Update3dCalc(std::unordered_map<ulong, float> &tmpNodesOutput, std::unordered_map<ulong, Neuron *> &tmpNeuronMap, NetworkGenome *nn) {
        mtx.lock();
        if (!update3dCalcLock) {
            update3dCalcLock = true;
            mtx.unlock();

            std::unordered_map<ulong, Color> nodId2Color;
            std::unordered_map<ulong, float> nodId2Size;

            for (auto &n: tmpNodesOutput) {
                if (n.second > 0.1f) {
                    if (tmpNeuronMap[n.first]->Layer == 0.f || tmpNeuronMap[n.first]->Layer == 1.f) {
                        nodId2Color[n.first] = WHITE;
                    } else {
                        nodId2Color[n.first] = YELLOW;
                    }
                    nodId2Size[n.first] = 0.1f * (n.second + 0.9f);
                } else {
                    if (tmpNeuronMap[n.first]->Layer == 0.f) {
                        nodId2Color[n.first] = BLUE;
                    } else if (tmpNeuronMap[n.first]->Layer == 1.f) {
                        nodId2Color[n.first] = RED;
                    } else {
                        nodId2Color[n.first] = GRAY;
                    }
                    nodId2Size[n.first] = 0.1f;
                }
            }

            std::vector<lineInfo> connectedNodesInfo;

            if (Opts.Enable3dCalc) {
                for (auto &c: nn->Connections) {
                    if (nodId2Size[c.ConnectedNeuronId[0]] > 0.1f && nodId2Size[c.ConnectedNeuronId[1]] > 0.1f && c.Enable && c.Weight > 0.f) {
                        connectedNodesInfo.push_back(lineInfo{c.ConnectedNeuronId[0], c.ConnectedNeuronId[1], c.Weight / Opts.WeightRange * 0.03f + 0.0001f, ColorAlpha(WHITE, 0.3f)});
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
    }

#endif

    std::vector<float> NeuralNetwork::FCNNFeedForwardPredict(NetworkGenome *nn, const std::vector<float> &inputs, bool isAccelerate) {
        std::vector<float> outputs(Opts.OutputSize);

        std::vector<float> neuronValues(nn->Neurons.size() - Opts.InputSize - Opts.OutputSize);
        ulong neuronCounter = 0;

        // 计算第一层隐藏节点的结果

        auto outputPerFirst = [&neuronValues, &nn, &inputs](uint fh, uint nc) {
            neuronValues[nc] = nn->Neurons[Opts.InputSize + Opts.OutputSize + fh].Bias;
            for (uint in = 0; in < Opts.InputSize; ++in) {
                neuronValues[nc] += inputs[in] * nn->Connections[Opts.InputSize * fh + in].Weight;
            }
            neuronValues[nc] = Opts.ActiveFunction(neuronValues[nc]);
        };

        if (isAccelerate) {
            for (uint fisrtHiddenLayer = 0; fisrtHiddenLayer < Opts.FCNN_hideLayers[0]; ++fisrtHiddenLayer) {
                tPool.push_task(outputPerFirst, fisrtHiddenLayer, neuronCounter);
                ++neuronCounter;
            }
            tPool.wait_for_tasks();
        } else {
            for (uint fisrtHiddenLayer = 0; fisrtHiddenLayer < Opts.FCNN_hideLayers[0]; ++fisrtHiddenLayer) {
                outputPerFirst(fisrtHiddenLayer, neuronCounter);
                ++neuronCounter;
            }
        }

        // 计算第二层及以后隐藏节点的结果

        ulong lastLayerConnectionCounter = Opts.FCNN_hideLayers[0] * Opts.InputSize;
        ulong lastLayerNeuronCounter = 0;

        auto outputPerHidden = [&neuronValues, &nn, &lastLayerNeuronCounter, &lastLayerConnectionCounter](uint hh, uint nc, uint hlh) {
            neuronValues[nc] = nn->Neurons[Opts.InputSize + Opts.OutputSize + lastLayerNeuronCounter + Opts.FCNN_hideLayers[hlh - 1] + hh].Bias;
            for (uint ln = 0; ln < Opts.FCNN_hideLayers[hlh - 1]; ++ln) {
                neuronValues[nc] += neuronValues[lastLayerNeuronCounter + ln] * nn->Connections[Opts.FCNN_hideLayers[hlh - 1] * hh + ln + lastLayerConnectionCounter].Weight;
            }
            neuronValues[nc] = Opts.ActiveFunction(neuronValues[nc]);
        };

        for (uint hl = 1; hl < Opts.FCNN_hideLayers.size(); ++hl) {
            if (isAccelerate) {
                for (uint otherHiddenLayer = 0; otherHiddenLayer < Opts.FCNN_hideLayers[hl]; ++otherHiddenLayer) {
                    tPool.push_task(outputPerHidden, otherHiddenLayer, neuronCounter, hl);
                    ++neuronCounter;
                }
                tPool.wait_for_tasks();
            } else {
                for (uint otherHiddenLayer = 0; otherHiddenLayer < Opts.FCNN_hideLayers[hl]; ++otherHiddenLayer) {
                    outputPerHidden(otherHiddenLayer, neuronCounter, hl);
                    ++neuronCounter;
                }
            }

            lastLayerNeuronCounter += Opts.FCNN_hideLayers[hl - 1];
            lastLayerConnectionCounter += Opts.FCNN_hideLayers[hl] * Opts.FCNN_hideLayers[hl - 1];
        }

        // 计算输出层节点的结果

        auto outputPerResult = [&nn, &outputs, &neuronValues, &lastLayerNeuronCounter, &lastLayerConnectionCounter](uint ol) {
            outputs[ol] = nn->Neurons[Opts.InputSize + ol].Bias;
            for (uint lastHiddenLayer = 0; lastHiddenLayer < Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1]; ++lastHiddenLayer) {
                outputs[ol] += neuronValues[lastLayerNeuronCounter + lastHiddenLayer] * nn->Connections[Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1] * ol + lastHiddenLayer + lastLayerConnectionCounter].Weight;
            }
            outputs[ol] = Opts.ActiveFunction(outputs[ol]);
        };

        if (isAccelerate) {
            for (uint ol = 0; ol < Opts.OutputSize; ++ol) {
                tPool.push_task(outputPerResult, ol);
            }
            tPool.wait_for_tasks();
        } else {
            for (uint ol = 0; ol < Opts.OutputSize; ++ol) {
                outputPerResult(ol);
            }
        }

#ifndef NO_3DNN

        if (Opts.Enable3dNN && Opts.Enable3dCalc) {
            std::unordered_map<ulong, Neuron *> tmpNeuronMap;  // 记录神经元id对应的神经元，需要的时候才能临时生成记录，不然神经元的数组push_back的新增内存的时候会改变原有地址
            for (auto &n: nn->Neurons) {
                tmpNeuronMap[n.Id] = &n;
            }

            std::unordered_map<ulong, float> tmpNodesOutput;

            for (ulong i = 0; i < Opts.InputSize; ++i) {
                tmpNodesOutput[i] = inputs[i];
            }

            for (ulong i = 0; i < Opts.OutputSize; ++i) {
                tmpNodesOutput[Opts.InputSize + i] = outputs[i];
            }

            for (ulong i = 0; i < neuronValues.size(); ++i) {
                tmpNodesOutput[Opts.InputSize + Opts.OutputSize + i] = neuronValues[i];
            }

            Update3dCalc(tmpNodesOutput, tmpNeuronMap, nn);
        }

#endif

        return outputs;
    }

    std::vector<float> NeuralNetwork::FeedForwardPredict(NetworkGenome *nn, const std::vector<float> &inputs, bool isAccelerate) {  // 在使用CPU多核运算时，由于多线程开销问题，神经网络结构太简单反而会运算得更慢
        std::vector<float> outputs(Opts.OutputSize);

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
        // [OPT][FIX] 原来所有对局部 map 的读写都锁全局 znn::mtx——tmpNodesOutput 是本函数的
        // 局部 map，与其它线程无关；并行评估不同 genome 的任务全部被全局锁串行化
        // （GA 多核基准实测慢 15 倍）。改为：非加速路径无需锁（调用内单线程）；加速路径用
        // 本函数局部的 std::mutex，只保护这个局部 map
        std::mutex localMtx;

        // [OPT] 预建"神经元→入边"倒排索引：原实现每个神经元都要全量扫描全部连接（O(N×C)），
        // 索引一次构建 O(C)，此后每个神经元只遍历自己的入边，总遍历 O(C)。
        // 稀疏 NEAT 网络 + 大种群场景下是数量级提升；稠密全连接网络复杂度持平
        std::unordered_map<ulong, std::vector<const Connection *>> incomingEdges;
        for (auto &connection: nn->Connections) {
            if (connection.Enable) {
                incomingEdges[connection.ConnectedNeuronId[1]].push_back(&connection);
            }
        }

        auto calculateNeuron = [&](ulong nid) {  // [OPT] std::function→auto闭包：去掉间接调用开销
            float thisOutput = 0.f;

            auto itEdges = incomingEdges.find(nid);
            if (itEdges != incomingEdges.end()) {
                for (auto *connection: itEdges->second) {
                    if (isAccelerate) {
                        // [FIX] 原代码在 shared_lock（读锁）内用 operator[] 读 tmpNodesOutput——
                        // operator[] 在 key 不存在时会插入（写操作），构成并发写 UB；改用 find 只读
                        std::unique_lock<std::mutex> lock(localMtx);
                        auto itOut = tmpNodesOutput.find(connection->ConnectedNeuronId[0]);
                        if (itOut != tmpNodesOutput.end()) {
                            thisOutput += itOut->second * connection->Weight;
                        }
                    } else {
                        thisOutput += tmpNodesOutput[connection->ConnectedNeuronId[0]] * connection->Weight;
                    }
                }
            }

            thisOutput = Opts.ActiveFunction(thisOutput + tmpNeuronMap[nid]->Bias);

            if (isAccelerate) {
                std::lock_guard<std::mutex> lock(localMtx);
                tmpNodesOutput[nid] = thisOutput;
            } else {
                tmpNodesOutput[nid] = thisOutput;
            }
        };

        uint i = 0;
        for (auto &n: tmpLayerMap[0.]) { // 初始化输入节点
            tmpNodesOutput[n->Id] = inputs[i];
            ++i;
        }

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

        if (Opts.Enable3dNN && Opts.Enable3dCalc) {
            Update3dCalc(tmpNodesOutput, tmpNeuronMap, nn);
        }

#endif

        return outputs;
    };

    std::vector<float> NeuralNetwork::BackPropagation(NetworkGenome *nn, const std::vector<float> &inputs, const std::vector<float> &wants, bool isAccelerate) {  // 如果当前预测fitness大于预设，则判断为解决问题，返回计算结果, 在使用CPU多核运算时，由于多线程开销问题，神经网络结构太简单反而会运算得更慢
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
        std::mutex localMtx;  // [OPT][FIX] 同 FeedForwardPredict：全局锁→局部锁

        // [OPT] 同 FeedForwardPredict：入边索引替代 O(N×C) 全连接扫描
        std::unordered_map<ulong, std::vector<const Connection *>> incomingEdges;
        for (auto &connection: nn->Connections) {
            if (connection.Enable) {
                incomingEdges[connection.ConnectedNeuronId[1]].push_back(&connection);
            }
        }

        auto calculateNeuron = [&](ulong nid) {  // [OPT] std::function→auto闭包
            float thisOutput = 0.f;

            auto itEdges = incomingEdges.find(nid);
            if (itEdges != incomingEdges.end()) {
                for (auto *connection: itEdges->second) {
                    if (isAccelerate) {
                        // [FIX] shared_lock 内 operator[] 会插入（写操作），并发写 UB；改用 find 只读
                        std::unique_lock<std::mutex> lock(localMtx);
                        auto itOut = tmpNodesOutput.find(connection->ConnectedNeuronId[0]);
                        if (itOut != tmpNodesOutput.end()) {
                            thisOutput += itOut->second * connection->Weight;
                        }
                    } else {
                        thisOutput += tmpNodesOutput[connection->ConnectedNeuronId[0]] * connection->Weight;
                    }
                }
            }

            float tmpNeuronBias = tmpNeuronMap[nid]->Bias;

            thisOutput = Opts.ActiveFunction(thisOutput + tmpNeuronBias);

            if (isAccelerate) {
                std::lock_guard<std::mutex> lock(localMtx);
                tmpNodesOutput[nid] = thisOutput;
            } else {
                tmpNodesOutput[nid] = thisOutput;
            }
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
        std::mutex errMtx;  // [OPT][FIX] 全局锁→局部锁（同上）

        // [OPT] "神经元→出边"倒排索引：误差回传同理，O(N×C)→O(C)
        std::unordered_map<ulong, std::vector<const Connection *>> outgoingEdges;
        for (auto &connection: nn->Connections) {
            if (connection.Enable) {
                outgoingEdges[connection.ConnectedNeuronId[0]].push_back(&connection);
            }
        }

        auto calculateNeuronError = [&](ulong nid) {  // [OPT] std::function→auto闭包
            float thisOutputError = 0.f;

            auto itEdges = outgoingEdges.find(nid);
            if (itEdges != outgoingEdges.end()) {
                for (auto *connection: itEdges->second) {
                    if (isAccelerate) {
                        // [FIX] shared_lock 内 operator[] 会插入（写操作），并发写 UB；改用 find 只读
                        std::unique_lock<std::mutex> lock(errMtx);
                        auto itErr = tmpNodesOutputError.find(connection->ConnectedNeuronId[1]);
                        if (itErr != tmpNodesOutputError.end()) {
                            thisOutputError += itErr->second * connection->Weight;
                        }
                    } else {
                        thisOutputError += tmpNodesOutputError[connection->ConnectedNeuronId[1]] * connection->Weight;
                    }
                }
            }

            // [FIX] 原式 DerivativeFunction(out) * out 多乘了一次输出值：
            // DerivativeSigmoid(fx)=fx*(1-fx) 已是 f'(out)，再乘 out 变成 out^2*(1-out)，
            // 梯度系统性偏小且失真（与输出层的写法也不一致）；去掉多余的 * tmpNodesOutput[nid]
            thisOutputError *= Opts.DerivativeFunction(tmpNodesOutput[nid]);

            // [OPT] 动量偏置更新 + 越界截断（原来是"越界拒绝"：偏置一旦到 ±BiasRange 边界
            // 整个更新被丢弃，学习在边界处永久卡死；clamp 允许边界处继续积累速度退回来）
            tmpNeuronMap[nid]->momentum = Opts.BPMomentum * tmpNeuronMap[nid]->momentum + Opts.LearnRate * thisOutputError;
            tmpNeuronMap[nid]->Bias = std::clamp(tmpNeuronMap[nid]->Bias + tmpNeuronMap[nid]->momentum, -Opts.BiasRange, Opts.BiasRange);

            if (isAccelerate) {
                std::lock_guard<std::mutex> lock(errMtx);
                tmpNodesOutputError[nid] = thisOutputError;
            } else {
                tmpNodesOutputError[nid] = thisOutputError;
            }
        };

        uint wantsCount = 0;
        for (std::map<double, std::vector<Neuron *>>::reverse_iterator ri = tmpLayerMap.rbegin(); ri != tmpLayerMap.rend(); ++ri) {
            for (auto &n: ri->second) {
                if (ri->first == 1.f) {   // 计算输出神经元点误差,更新神经元偏置
                    tmpNodesOutputError[n->Id] = Opts.DerivativeFunction(tmpNodesOutput[n->Id]) * (wants[wantsCount] - tmpNodesOutput[n->Id]);
                    ++wantsCount;

                    // [OPT] 动量 + 截断（同 calculateNeuronError：拒绝→clamp）
                    n->momentum = Opts.BPMomentum * n->momentum + Opts.LearnRate * tmpNodesOutputError[n->Id];
                    n->Bias = std::clamp(n->Bias + n->momentum, -Opts.BiasRange, Opts.BiasRange);

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
            // [FIX] 原代码对禁用连接也更新权重：禁用连接的权重被 BP 污染后，一旦变异重新启用
            // （MutateAddConnection 会复用并 Enable=true），等于把陈旧错误权重带回网络；
            // 与 forward/误差计算只遍历 Enable 连接的逻辑保持一致，跳过禁用连接
            if (!connection.Enable) {
                continue;
            }

            // [OPT] 动量权重更新 + 越界截断（同上：原来越界整个更新被丢弃，±WeightRange 边界卡死）
            connection.momentum = Opts.BPMomentum * connection.momentum + Opts.LearnRate * tmpNodesOutputError[connection.ConnectedNeuronId[1]] * tmpNodesOutput[connection.ConnectedNeuronId[0]];
            connection.Weight = std::clamp(connection.Weight + connection.momentum, -Opts.WeightRange, Opts.WeightRange);
        }

        //        nn->Age = 0;

        if (outputs.size() != wants.size()) {
            std::cerr << "BackPropagation: outputs.size(" << outputs.size() << ") != wants.size(" << wants.size() << ")\n";
            exit(0);
        }

        return outputs;
    };

    std::vector<float> NeuralNetwork::FCNNBackPropagation(NetworkGenome *nn, const std::vector<float> &inputs, const std::vector<float> &wants, bool isAccelerate) {
        std::vector<float> outputs(Opts.OutputSize);

        std::vector<float> neuronValues(nn->Neurons.size() - Opts.InputSize - Opts.OutputSize);
        ulong neuronCounter = 0;

        auto outputPerFirst = [&neuronValues, &nn, &inputs](uint fh, uint nc) {
            neuronValues[nc] = nn->Neurons[Opts.InputSize + Opts.OutputSize + fh].Bias;
            for (uint in = 0; in < Opts.InputSize; ++in) {
                neuronValues[nc] += inputs[in] * nn->Connections[Opts.InputSize * fh + in].Weight;
            }
            neuronValues[nc] = Opts.ActiveFunction(neuronValues[nc]);
        };

        if (isAccelerate) {
            for (uint fisrtHiddenLayer = 0; fisrtHiddenLayer < Opts.FCNN_hideLayers[0]; ++fisrtHiddenLayer) {
                tPool.push_task(outputPerFirst, fisrtHiddenLayer, neuronCounter);
                ++neuronCounter;
            }
            tPool.wait_for_tasks();
        } else {
            for (uint fisrtHiddenLayer = 0; fisrtHiddenLayer < Opts.FCNN_hideLayers[0]; ++fisrtHiddenLayer) {
                outputPerFirst(fisrtHiddenLayer, neuronCounter);
                ++neuronCounter;
            }
        }

        ulong lastLayerConnectionCounter = Opts.FCNN_hideLayers[0] * Opts.InputSize;
        ulong lastLayerNeuronCounter = 0;

        auto outputPerHidden = [&neuronValues, &nn, &lastLayerNeuronCounter, &lastLayerConnectionCounter](uint hh, uint nc, uint hlh) {
            neuronValues[nc] = nn->Neurons[Opts.InputSize + Opts.OutputSize + lastLayerNeuronCounter + Opts.FCNN_hideLayers[hlh - 1] + hh].Bias;
            for (uint ln = 0; ln < Opts.FCNN_hideLayers[hlh - 1]; ++ln) {
                neuronValues[nc] += neuronValues[lastLayerNeuronCounter + ln] * nn->Connections[Opts.FCNN_hideLayers[hlh - 1] * hh + ln + lastLayerConnectionCounter].Weight;
            }
            neuronValues[nc] = Opts.ActiveFunction(neuronValues[nc]);
        };

        for (uint hl = 1; hl < Opts.FCNN_hideLayers.size(); ++hl) {
            if (isAccelerate) {
                for (uint otherHiddenLayer = 0; otherHiddenLayer < Opts.FCNN_hideLayers[hl]; ++otherHiddenLayer) {
                    tPool.push_task(outputPerHidden, otherHiddenLayer, neuronCounter, hl);
                    ++neuronCounter;
                }
                tPool.wait_for_tasks();
            } else {
                for (uint otherHiddenLayer = 0; otherHiddenLayer < Opts.FCNN_hideLayers[hl]; ++otherHiddenLayer) {
                    outputPerHidden(otherHiddenLayer, neuronCounter, hl);
                    ++neuronCounter;
                }
            }

            lastLayerNeuronCounter += Opts.FCNN_hideLayers[hl - 1];
            lastLayerConnectionCounter += Opts.FCNN_hideLayers[hl] * Opts.FCNN_hideLayers[hl - 1];
        }

        auto outputPerResult = [&nn, &outputs, &neuronValues, &lastLayerNeuronCounter, &lastLayerConnectionCounter](uint ol) {
            outputs[ol] = nn->Neurons[Opts.InputSize + ol].Bias;
            for (uint lastHiddenLayer = 0; lastHiddenLayer < Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1]; ++lastHiddenLayer) {
                outputs[ol] += neuronValues[lastLayerNeuronCounter + lastHiddenLayer] * nn->Connections[Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1] * ol + lastHiddenLayer + lastLayerConnectionCounter].Weight;
            }
            outputs[ol] = Opts.ActiveFunction(outputs[ol]);
        };

        if (isAccelerate) {
            for (uint ol = 0; ol < Opts.OutputSize; ++ol) {
                tPool.push_task(outputPerResult, ol);
            }
            tPool.wait_for_tasks();
        } else {
            for (uint ol = 0; ol < Opts.OutputSize; ++ol) {
                outputPerResult(ol);
            }
        }

        // 上面是正向计算全部节点输出,接下来开始反向传播
        // 先计算每个节点的误差,偏导数又称为误差项也称为灵敏度

        std::vector<float> neuronErrors(nn->Neurons.size() - Opts.InputSize - Opts.OutputSize, 0.f);
        std::vector<float> resultErrors(Opts.OutputSize);

        ulong neuronErrorCounter = std::accumulate(Opts.FCNN_hideLayers.begin(), Opts.FCNN_hideLayers.end(), -Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1]);
        ulong connectionErrorCounter = nn->Connections.size() - Opts.OutputSize * Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1];

        // 计算输出节点的误差,更新bias

        for (uint oi = 0; oi < Opts.OutputSize; ++oi) {
            resultErrors[oi] = Opts.DerivativeFunction(outputs[oi]) * (wants[oi] - outputs[oi]);

            // 更新输出节点的偏置  [OPT] 动量 + 越界截断（拒绝→clamp，同 BackPropagation）
            nn->Neurons[Opts.InputSize + oi].momentum = Opts.BPMomentum * nn->Neurons[Opts.InputSize + oi].momentum + Opts.LearnRate * resultErrors[oi];
            nn->Neurons[Opts.InputSize + oi].Bias = std::clamp(nn->Neurons[Opts.InputSize + oi].Bias + nn->Neurons[Opts.InputSize + oi].momentum, -Opts.BiasRange, Opts.BiasRange);

            // 计算最后一层隐藏节点的误差
            for (uint lastHiddenLayerNeuron = 0; lastHiddenLayerNeuron < Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1]; ++lastHiddenLayerNeuron) {
                neuronErrors[neuronErrorCounter + lastHiddenLayerNeuron] += resultErrors[oi] * nn->Connections[Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1] * oi + lastHiddenLayerNeuron + connectionErrorCounter].Weight;
            }
        }

        // 计算最后一层隐藏层至第一层隐藏层节点以后的误差
        for (uint hl = Opts.FCNN_hideLayers.size() - 1; hl > 0; --hl) {
            for (uint thisHiddenLayerNeuron = 0; thisHiddenLayerNeuron < Opts.FCNN_hideLayers[hl]; ++thisHiddenLayerNeuron) {
                // [FIX] 原式多乘了一次 neuronValues（同 BackPropagation 的导数问题）：去掉多余的 * neuronValues[...]
                neuronErrors[neuronErrorCounter + thisHiddenLayerNeuron] *= Opts.DerivativeFunction(neuronValues[neuronErrorCounter + thisHiddenLayerNeuron]);
            }

            neuronErrorCounter -= Opts.FCNN_hideLayers[hl - 1];
            connectionErrorCounter -= Opts.FCNN_hideLayers[hl] * Opts.FCNN_hideLayers[hl - 1];

            for (uint thisHiddenLayer = 0; thisHiddenLayer < Opts.FCNN_hideLayers[hl]; ++thisHiddenLayer) {
                for (uint lastHiddenLayer = 0; lastHiddenLayer < Opts.FCNN_hideLayers[hl - 1]; ++lastHiddenLayer) {
                    neuronErrors[neuronErrorCounter + lastHiddenLayer] += neuronErrors[neuronErrorCounter + thisHiddenLayer + Opts.FCNN_hideLayers[hl - 1]] * nn->Connections[connectionErrorCounter + Opts.FCNN_hideLayers[hl - 1] * thisHiddenLayer + lastHiddenLayer].Weight;
                }
            }
        }

        // 计算第一层隐藏层节点的误差
        for (uint firstHiddenLayerNeuron = 0; firstHiddenLayerNeuron < Opts.FCNN_hideLayers[0]; ++firstHiddenLayerNeuron) {
            // [FIX] 原式多乘了一次 neuronValues（同上）；去掉多余的 * neuronValues[...]
            neuronErrors[firstHiddenLayerNeuron] *= Opts.DerivativeFunction(neuronValues[firstHiddenLayerNeuron]);
        }

        // 更新连接权重

        // 更新第一二层节点间连接权重  [OPT] 动量 + 越界截断（拒绝→clamp）
        for (uint firstHiddenLayer = 0; firstHiddenLayer < Opts.FCNN_hideLayers[0]; ++firstHiddenLayer) {
            for (uint in = 0; in < Opts.InputSize; ++in) {
                auto &conn = nn->Connections[Opts.InputSize * firstHiddenLayer + in];
                conn.momentum = Opts.BPMomentum * conn.momentum + Opts.LearnRate * neuronErrors[firstHiddenLayer] * inputs[in];
                conn.Weight = std::clamp(conn.Weight + conn.momentum, -Opts.WeightRange, Opts.WeightRange);
            }
        }

        // 更新第二层至最后一层节点间连接权重
        lastLayerConnectionCounter = Opts.FCNN_hideLayers[0] * Opts.InputSize;
        lastLayerNeuronCounter = Opts.FCNN_hideLayers[0];

        for (uint hl = 1; hl < Opts.FCNN_hideLayers.size(); ++hl) {
            for (uint otherHiddenLayer = 0; otherHiddenLayer < Opts.FCNN_hideLayers[hl]; ++otherHiddenLayer) {
                for (uint ln = 0; ln < Opts.FCNN_hideLayers[hl - 1]; ++ln) {
                    auto &conn = nn->Connections[Opts.FCNN_hideLayers[hl - 1] * otherHiddenLayer + ln + lastLayerConnectionCounter];
                    conn.momentum = Opts.BPMomentum * conn.momentum + Opts.LearnRate * neuronErrors[lastLayerNeuronCounter + otherHiddenLayer] * neuronValues[ln + lastLayerNeuronCounter - Opts.FCNN_hideLayers[hl - 1]];
                    conn.Weight = std::clamp(conn.Weight + conn.momentum, -Opts.WeightRange, Opts.WeightRange);
                }
            }
            lastLayerNeuronCounter += Opts.FCNN_hideLayers[hl];
            lastLayerConnectionCounter += Opts.FCNN_hideLayers[hl] * Opts.FCNN_hideLayers[hl - 1];
        }

//        std::cout << "lastLayerNeuronCounter "<< lastLayerNeuronCounter << "\n";

        // 更新输出层节点的连接权重  [OPT] 动量 + 越界截断（拒绝→clamp）
        for (uint ol = 0; ol < Opts.OutputSize; ++ol) {
            for (uint lastHiddenLayer = 0; lastHiddenLayer < Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1]; ++lastHiddenLayer) {
                auto &conn = nn->Connections[Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1] * ol + lastHiddenLayer + lastLayerConnectionCounter];
                conn.momentum = Opts.BPMomentum * conn.momentum + Opts.LearnRate * resultErrors[ol] * neuronValues[lastLayerNeuronCounter - Opts.FCNN_hideLayers[Opts.FCNN_hideLayers.size() - 1] + lastHiddenLayer];
                conn.Weight = std::clamp(conn.Weight + conn.momentum, -Opts.WeightRange, Opts.WeightRange);
            }
        }

        // 更新隐藏层节点偏置  [OPT] 动量 + 越界截断（拒绝→clamp）
        for (uint ni = 0; ni < neuronErrors.size(); ++ni) {
            nn->Neurons[Opts.InputSize + Opts.OutputSize + ni].momentum = Opts.BPMomentum * nn->Neurons[Opts.InputSize + Opts.OutputSize + ni].momentum + Opts.LearnRate * neuronErrors[ni];
            nn->Neurons[Opts.InputSize + Opts.OutputSize + ni].Bias = std::clamp(nn->Neurons[Opts.InputSize + Opts.OutputSize + ni].Bias + nn->Neurons[Opts.InputSize + Opts.OutputSize + ni].momentum, -Opts.BiasRange, Opts.BiasRange);
        }

        if (outputs.size() != wants.size()) {
            std::cerr << "BackPropagation: outputs.size(" << outputs.size() << ") != wants.size(" << wants.size() << ")\n";
            exit(0);
        }

        return outputs;
    }

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
        std::string data = std::to_string(Opts.InputSize) + ",";

        for (auto hl : Opts.FCNN_hideLayers) {
            data += std::to_string(hl) + ",";
        }

        data += std::to_string(Opts.OutputSize) + "\n~\n";

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
                if (datas.size() < 2) {
                    std::cerr << "datas.size() < 2";
                    exit(0);
                }

                if (datas.size() > 2) {
                    Opts.FCNN_hideLayers.clear();
                    for (uint hl = 1; hl < datas.size() - 1; ++hl) {
                        Opts.FCNN_hideLayers.push_back(std::stoul(datas[hl]));
                    }
                }

                InputSize = std::stoul(datas[0]);
                OutputSize = std::stoul(datas[datas.size() - 1]);
                continue;
            }

            if (dataType == 1) {
                // [FIX] 手改过的 .nn 文件可能有空行/缺字段，直接 datas[1]/datas[2] 会越界 UB；先检查再访问
                auto datas = SplitString(line, ",");
                if (datas.size() < 3) {
                    std::cerr << "ImportNN wrong neuron line: \"" << line << "\"\n";
                    exit(0);
                }
                double nnLayer = std::stod(datas[1]);
                ulong id = std::stoul(datas[0]);
                newNeurons.push_back(Neuron{.Id = id, .Bias = std::stof(datas[2]), .Layer = nnLayer,});
                tmpIdMap[id] = 0;
                continue;
            }

            // [FIX] 连接段同理：空行/畸形行先跳过或报错，避免 datas[0]/cnid[1] 越界 UB
            auto datas = SplitString(line, ",");
            if (datas.size() < 2) {
                std::cerr << "ImportNN wrong connection line: \"" << line << "\"\n";
                exit(0);
            }
            auto cnid = SplitString(datas[0], "->");
            if (cnid.size() < 2) {
                std::cerr << "ImportNN wrong connection endpoints: \"" << line << "\"\n";
                exit(0);
            }
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
                // [FIX] 原公式用“导入时的 OutputSize/InputSize”拼新 Id，与 MutateAddNeuron 生成
                // 新 Id 的规则（当前 Opts 尺寸）不一致，混合导入/续训时可能与其他神经元 Id 碰撞；
                // 改为与 MutateAddNeuron 相同的公式：HiddenNeuronInnovations.size() + 当前InputSize + 当前OutputSize，
                // 并加上偏移 i 保证补充的多个神经元之间互不冲突
                ulong newId = HiddenNeuronInnovations.size() + Opts.InputSize + Opts.OutputSize + FCHidenNeuronSize + i;
                newNeurons.push_back(Neuron{.Id = newId, .Bias = 0.f, .Layer = 0.,});
            }
        }

        if (Opts.OutputSize > OutputSize) {
            for (ulong i = 0; i < Opts.OutputSize - OutputSize; ++i) {
                // [FIX] 同上：与 MutateAddNeuron 公式保持一致 + 输入补充量偏移，避免 Id 碰撞
                ulong newId = HiddenNeuronInnovations.size() + Opts.InputSize + Opts.OutputSize + FCHidenNeuronSize
                              + (Opts.InputSize - InputSize) + i;
                newNeurons.push_back(Neuron{.Id = newId, .Bias =
                float(random() % long(Opts.NewNNBiasRange * 200) - long(Opts.NewNNBiasRange * 100)) / 100.f, .Layer = 1.,});
            }
        }

        return NetworkGenome{.Neurons = newNeurons, .Connections = newConnections,};
    }

    void NeuralNetwork::ExportInnovations(std::string fileName) {
        std::string data;

        if (Opts.StartWithFCNN) {
            for (auto &hl : Opts.FCNN_hideLayers) {
                data += std::to_string(hl) + "\n";
            }
        }

        data += "~\n";

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

        int dataType = 0;
        uint hl = 0;
        while (getline(input_file, line)) {
            if (line == "~") {
                ++dataType;
                continue;
            }

            if (dataType == 0) {
                // [FIX] 原代码直接 Opts.FCNN_hideLayers[hl]，若 .innov 文件带 FC 头但当前
                // Opts.FCNN_hideLayers 为空或长度不匹配，会越界；先检查再访问
                if (hl >= Opts.FCNN_hideLayers.size()) {
                    std::cerr << "Innovations FCNN header longer than Opts.FCNN_hideLayers\n";
                    exit(0);
                }
                if (Opts.FCNN_hideLayers[hl] != std::stoi(line)) {
                    std::cerr << "Innovations based FCNN wrong\n";
                    exit(0);
                }
                ++hl;
            }

            if (dataType == 1) {
                auto datas = SplitString(line, ",");
                //                HiddenNeuronInnovations.insert({{std::stoul(datas[0]), std::stoul(datas[1])}, std::stoul(datas[2])});
                HiddenNeuronInnovations[{std::stoul(datas[0]), std::stoul(datas[1])}] = std::stoul(datas[2]);
            }
        }
    }
}

#endif //MYNEAT_NERAULNETWORK_HPP

