#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include "lib/SimpleNEAT.hpp"


int main() {
    // Window
    sf::RenderWindow window(sf::VideoMode(800, 800), "path findder", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(120);
    sf::Event ev{};
    bool isKeepLeft = false;
    std::map<std::vector<long>, sf::RectangleShape> blocks;
    std::map<std::vector<long>, int> linePos;

    std::vector<long> beginPos = {30, 30};
    std::vector<long> endPos = {760, 760};
    static float bestDistance = std::sqrt(float(std::pow(beginPos[0] - endPos[0], 2) + std::pow(beginPos[1] - endPos[1], 2)));


    struct Player {
        std::map<std::vector<long>, int> passedPath;
        std::vector<long> statusPos;
        bool isDone = false;
        uint stepCount = 0;
        float distanceLeft = bestDistance;
    };

    std::map<znn::NetworkGenome *, Player> players;
    Player Champion;

    bool isStart = false;
    bool isAllDie = false;
    bool isCrow = false;
    bool canDrawChampion = false;

    std::function<bool(Player &, std::vector<long> &)> isCrash = [&linePos, &endPos](Player &player, std::vector<long> &nextMove) {
        if (nextMove[0] < 0 || nextMove[0] >= 800 || nextMove[1] < 0 || nextMove[1] >= 800 || player.passedPath.contains(nextMove) || linePos.contains(nextMove) || nextMove == endPos) {
            return true;
        }
        return false;
    };

    for (long i = 0; i < 80; ++i) {
        for (long ii = 0; ii < 80; ++ii) {
            sf::RectangleShape box(sf::Vector2f(10.f, 10.f));
            box.setPosition(float(i * 10), float(ii * 10));
            box.setFillColor(sf::Color(0, 0, 0, 255));
            blocks[{i * 10, ii * 10}] = box;
        }
    }

    /*
     * 0 1 2
     * 3   4
     * 5 6 7
     */

    std::function<std::vector<float>(std::vector<long> &)> getAround = [&linePos](std::vector<long> &status) {
        std::vector<float> around = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (linePos.contains({status[0] - 10, status[1] - 10}) || status[0] - 10 < 0 || status[1] - 10 < 0) {
            around[0] = 1.f;
        }
        if (linePos.contains({status[0], status[1] - 10}) || status[1] - 10 < 0) {
            around[1] = 1.f;
        }
        if (linePos.contains({status[0] + 10, status[1] - 10}) || status[0] + 10 >= 800 || status[1] - 10 < 0) {
            around[2] = 1.f;
        }
        if (linePos.contains({status[0] - 10, status[1]}) || status[0] - 10 < 0) {
            around[3] = 1.f;
        }
        if (linePos.contains({status[0] + 10, status[1]}) || status[0] + 10 >= 800) {
            around[4] = 1.f;
        }
        if (linePos.contains({status[0] - 10, status[1] + 10}) || status[0] - 10 < 0 || status[1] + 10 >= 800) {
            around[5] = 1.f;
        }
        if (linePos.contains({status[0], status[1] + 10}) || status[1] + 10 >= 800) {
            around[6] = 1.f;
        }
        if (linePos.contains({status[0] + 10, status[1] + 10}) || status[0] + 10 >= 800 || status[1] + 10 >= 800) {
            around[7] = 1.f;
        }
        return around;
    };

    std::function<std::vector<long>(std::vector<long> &, uint &)> getNextMove = [](std::vector<long> &status, uint &action) {
        switch (action) {
            case 0:
                return std::vector<long>{status[0] - 10, status[1] - 10};
            case 1:
                return std::vector<long>{status[0], status[1] - 10};
            case 2:
                return std::vector<long>{status[0] + 10, status[1] - 10};
            case 3:
                return std::vector<long>{status[0] - 10, status[1]};
            case 4:
                return std::vector<long>{status[0] + 10, status[1]};
            case 5:
                return std::vector<long>{status[0] - 10, status[1] + 10};
            case 6:
                return std::vector<long>{status[0], status[1] + 10};
            case 7:
                return std::vector<long>{status[0] + 10, status[1] + 10};
        }
    };

    znn::SimpleNeat sneat;

    auto createPop = [&]() {  // 构建基因种群
        znn::Opts.InputSize = 12;
        znn::Opts.OutputSize = 8;
        znn::Opts.ActiveFunction = znn::Sigmoid;
        znn::Opts.IterationTimes = 900;
        znn::Opts.FitnessThreshold = 0.f;
        znn::Opts.IterationCheckPoint = 0;
        znn::Opts.ThreadCount = 16;
        znn::Opts.MutateAddNeuronRate = 0.0001f;
        znn::Opts.MutateAddConnectionRate = 0.99f;
        znn::Opts.PopulationSize = 300;
        znn::Opts.ChampionKeepSize = 30;
        znn::Opts.NewSize = 20;
        znn::Opts.KeepWorstSize = 30;
        znn::Opts.ChampionToNewSize = 90;
        znn::Opts.KeepComplexSize = 1;
        znn::Opts.WeightRange = 32.f;
        znn::Opts.BiasRange = 16.f;
        znn::Opts.MutateWeightDirectOrNear = 0.5f;
        znn::Opts.MutateWeightNearRange = 6;
        znn::Opts.MutateBiasDirectOrNear = 0.5f;
        znn::Opts.MutateWeightNearRange = 6;
        znn::Opts.Enable3dNN = true;

        sneat.StartNew();

        isCrow = true;
    };

    auto initPlayers = [&]() {
        for (auto &g: sneat.population.NeuralNetworks) {
            Player player = Player{};
            player.statusPos = beginPos;
            player.passedPath[beginPos] = 0;
            players[&g] = player;
        }
    };

    uint rounds = 1;
    uint realRounds = 1;

    auto getFitness = [&]() {
        std::map<znn::NetworkGenome *, float> popFitness;
        for (auto &p: players) {
            popFitness[p.first] = (bestDistance - p.second.distanceLeft) * 0.5f + float(p.second.stepCount) * 0.5f;  // TODO 分数评判有问题
            if (p.second.distanceLeft < 15) {
                popFitness[p.first] = popFitness[p.first] / float(p.second.stepCount) + bestDistance;
            }
        }
        return popFitness;
    };

    auto drawChampion = [&]() {
//        std::cout << p.distanceLeft << "\n";
        for (auto &pp: Champion.passedPath) {
            blocks[pp.first].setFillColor(sf::Color(200, 200, 255, 200));
            window.draw(blocks[pp.first]);
        }

        blocks[Champion.statusPos].setFillColor(sf::Color(250, 0, 0, 255));

        window.draw(blocks[Champion.statusPos]);
    };

    auto singleFromLoop = [&]() {
        using namespace znn;
        auto populationFitness = getFitness();
        auto orderedPopulation = sneat.OrderByFitness(populationFitness);
        auto orderedByComplex = sneat.OrderByComplex();

        Champion = players[orderedPopulation[0]];

        std::cout << "gen: " << realRounds << " " << orderedPopulation[0] << " " << orderedPopulation[0]->Neurons.size() << " " << orderedPopulation[0]->Connections.size() << " steps: "
                  << players[orderedPopulation[0]].stepCount << " distance left: " << players[orderedPopulation[0]].distanceLeft << " fitness: " << populationFitness[orderedPopulation[0]]
                  << " most complex: " << orderedByComplex[0]->Neurons.size() << std::endl;
        ++realRounds;

        if (rounds >= Opts.IterationTimes) {
            auto simplifiedBestNN = sneat.population.generation.neuralNetwork.SimplifyRemoveDisable(*orderedPopulation[0]);
            auto compressedLeftBestNN = sneat.population.generation.neuralNetwork.SimplifyRemoveUselessConnectionLeft(simplifiedBestNN);
            auto compressedRightBestNN = sneat.population.generation.neuralNetwork.SimplifyRemoveUselessConnectionRight(compressedLeftBestNN);
            sneat.population.generation.neuralNetwork.ExportNN(compressedRightBestNN, "./champion");
            isStart = false;
            isAllDie = false;
            canDrawChampion = true;
            rounds = 1;
            std::cout << "Training done.\n";
            return;
        }

        if (players[orderedPopulation[0]].distanceLeft < 15) {
            canDrawChampion = true;
        }

        std::vector<NetworkGenome> tmpPopulation(Opts.PopulationSize);
        std::vector<std::future<void>> thisFuture;  // 如果用这个线程池的push_task函数，后面需要wait_for_tasks()，会卡死

        uint indexOutside = 0;
        for (auto &nn: tmpPopulation) {
            thisFuture.push_back(tPool.submit([&]() {
                mtx.lock();
                uint index = indexOutside;
                ++indexOutside;
                mtx.unlock();

                if (index < Opts.ChampionToNewSize) {
                    nn = *orderedPopulation[index % Opts.ChampionKeepSize];  // 选取ChampionKeepSize个个体填满前ChampionToNewSize个
                    if (index >= Opts.ChampionKeepSize) {
                        sneat.population.generation.MutateNetworkGenome(nn);  // 除开原始冠军，他们的克隆体进行变异
                    }
                } else if (index < Opts.PopulationSize - Opts.NewSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                    auto nn0 = orderedPopulation[random() % Opts.ChampionKeepSize];
                    auto nn1 = orderedPopulation[Opts.ChampionKeepSize + random() % (Opts.PopulationSize - Opts.ChampionKeepSize)];
                    nn = sneat.population.generation.GetChildByCrossing(nn0, nn1);
                    if ((index % 2 == 0 || nn0 == nn1) && nn0->Neurons.size() < orderedByComplex[0]->Neurons.size() && nn1->Neurons.size() < orderedByComplex[0]->Neurons.size()) {
                        sneat.population.generation.MutateNetworkGenome(nn);  // 繁殖以后进行变异
                    }
                    ++index;
                } else if (index < Opts.PopulationSize - Opts.KeepWorstSize - Opts.KeepComplexSize) {
                    nn = sneat.population.generation.neuralNetwork.NewNN();
                } else if (index < Opts.PopulationSize - Opts.KeepWorstSize) {
                    nn = *orderedByComplex[index % Opts.KeepComplexSize];
                    sneat.population.generation.EnableAllConnections(nn);
                } else {
                    nn = *orderedPopulation[index];
                    sneat.population.generation.MutateNetworkGenome(nn);
                }
            }));
        }
        //        exit(0);

        for (auto &f: thisFuture) {
            f.wait();
        }

        sneat.population.NeuralNetworks = tmpPopulation;

        initPlayers();

        ++rounds;
    };

    //Gmae Loop
    for (;;) {
        if (!isStart) {
            while (window.pollEvent(ev)) {
                switch (ev.type) {
                    case sf::Event::Closed:
                        window.close();
                        exit(0);
                    case sf::Event::KeyPressed:
                        if (ev.key.code == sf::Keyboard::Space) {
                            std::cout << "Start training\n";
                            isAllDie = false;
                            isStart = true;
                            canDrawChampion = false;
                            if (!isCrow) {
                                createPop();
                            }
                            initPlayers();
                            break;
                        }
                        if (ev.key.code == sf::Keyboard::Escape) {
                            isCrow = false;
                            realRounds = 0;
                            std::cout << "Clear.\n";
                            players.clear();
                            canDrawChampion = false;
                            window.clear(sf::Color(0, 0, 0, 255)); // Clear old frame
                            break;
                        }
                        {
                            auto thisMouse = sf::Mouse::getPosition(window);
                            if (!isCrow && ev.key.code == sf::Keyboard::Num1) {
                                beginPos = {thisMouse.x / 10 * 10, thisMouse.y / 10 * 10};
                                bestDistance = std::sqrt(float(std::pow(beginPos[0] - endPos[0], 2) + std::pow(beginPos[1] - endPos[1], 2)));
                                std::cout << "begin set: " << beginPos[0] << " " << beginPos[1] << "\n";
                            }
                            if (!isCrow && ev.key.code == sf::Keyboard::Num2) {
                                endPos = {thisMouse.x / 10 * 10, thisMouse.y / 10 * 10};
                                bestDistance = std::sqrt(float(std::pow(beginPos[0] - endPos[0], 2) + std::pow(beginPos[1] - endPos[1], 2)));
                                std::cout << "end set: " << endPos[0] << " " << endPos[1] << "\n";
                            }
                            window.clear(sf::Color(0, 0, 0, 255)); // Clear old frame
                            break;
                        }
                        break;
                    case sf::Event::MouseButtonPressed:
                        if (ev.mouseButton.button == sf::Mouse::Left) {
                            isKeepLeft = true;
                        }
                        if (ev.mouseButton.button == sf::Mouse::Right) {
                            linePos.clear();
                            window.clear(sf::Color(0, 0, 0, 255)); // Clear old frame
                        }
                        break;
                    case sf::Event::MouseButtonReleased:
                        if (ev.mouseButton.button == sf::Mouse::Left) {
                            isKeepLeft = false;
                        }
                        break;
                }
            }
        } else {
            while (window.pollEvent(ev)) {
                switch (ev.type) {
                    case sf::Event::Closed:
                        window.close();
                        exit(0);
                    case sf::Event::KeyPressed:
                        if (ev.key.code == sf::Keyboard::Space) {
                            isStart = false;
                            isAllDie = false;
                            canDrawChampion = true;
                            rounds = 1;
                            std::cout << "Training cancel.\n";
                            break;
                        }
                }
            }
        }

        // Update
        if (isKeepLeft && !isStart) {
            auto pos = sf::Mouse::getPosition(window);
            linePos[{pos.x / 10 * 10, pos.y / 10 * 10}] = 0;
            linePos[{pos.x / 10 * 10 - 10, pos.y / 10 * 10}] = 0;
            linePos[{pos.x / 10 * 10, pos.y / 10 * 10 - 10}] = 0;
            linePos[{pos.x / 10 * 10 - 10, pos.y / 10 * 10 - 10}] = 0;
        }

        if (isStart && !isAllDie) {
            isAllDie = true;
            for (auto &p: players) {
                if (!p.second.isDone) {
                    std::vector<float> around = getAround(p.second.statusPos);
                    auto predictActions = sneat.population.generation.neuralNetwork.FeedForwardPredict(p.first, {float(p.second.statusPos[0]) / 800.f, float(p.second.statusPos[1]) / 800.f,
                                                                                                                 float(endPos[0]) / 800.f, float(endPos[1]) / 800.f, around[0], around[1], around[2],
                                                                                                                 around[3], around[4], around[5], around[6], around[7]}, false);
                    std::map<float, uint> actions;
                    for (uint i = 0; i < 8; ++i) {
                        actions[predictActions[i]] = i;
                    }
                    std::sort(predictActions.begin(), predictActions.end());
                    uint choseAction = actions[predictActions[predictActions.size() - 1]];
                    std::vector<long> nextMove = getNextMove(p.second.statusPos, choseAction);
                    if (isCrash(p.second, nextMove)) {
                        p.second.isDone = true;
                        p.second.distanceLeft = std::sqrt(std::pow(p.second.statusPos[0] - endPos[0], 2) + std::pow(p.second.statusPos[1] - endPos[1], 2));
                        //                        std::cout << &p.second << " died step: " << p.second.stepCount << " left: " << p.second.distanceLeft << "\n";
                        if (nextMove[0] < 0 || nextMove[0] >= 800 || nextMove[1] < 0 || nextMove[1] >= 800 || linePos.contains(nextMove)) {
                            p.second.distanceLeft += 50.f;
                        }
                    } else {
                        ++p.second.stepCount;
                        p.second.statusPos = nextMove;
                        p.second.passedPath[nextMove] = 0;
                        isAllDie = false;
                    }
                }
            }
        }

        if (isAllDie && isStart) {
            isAllDie = false;
//            std::cout << "All died\n";
            singleFromLoop();
        }

        if (!isStart || rounds % 10 == 0 || canDrawChampion || isAllDie) {
            if (canDrawChampion && (rounds % 10 == 1 || !isStart)) {
                window.clear(sf::Color(0, 0, 0, 255)); // Clear old frame
                drawChampion();
            } else if ((isStart || isAllDie) && rounds % 10 == 0) {
                window.clear(sf::Color(0, 0, 0, 255)); // Clear old frame

                std::vector<std::vector<long>> tmpDiedPoints;
                for (auto &p: players) {
                    for (auto &pp: p.second.passedPath) {
                        blocks[pp.first].setFillColor(sf::Color(50, 50, 200, 60));
                        window.draw(blocks[pp.first]);
                    }
                    if (p.second.isDone) {
                        tmpDiedPoints.push_back(p.second.statusPos);
                    } else {
                        blocks[p.second.statusPos].setFillColor(sf::Color(0, 200, 0, 100));
                    }
                    window.draw(blocks[p.second.statusPos]);
                }

                for (auto &dp: tmpDiedPoints) {
                    blocks[dp].setFillColor(sf::Color(250, 0, 0, 255));
                    window.draw(blocks[dp]);
                }
//                }
            }

            for (auto &lp: linePos) {
                blocks[lp.first].setFillColor(sf::Color(255, 255, 255, 255));
                window.draw(blocks[lp.first]);
            }

            blocks[beginPos].setFillColor(sf::Color(255, 255, 0, 255));
            window.draw(blocks[beginPos]);

            blocks[endPos].setFillColor(sf::Color(0, 150, 255, 255));
            window.draw(blocks[endPos]);

            // Draw game

            window.display(); // Tell app window is done drawing
        }

        if (isStart) {
            std::thread update3d(znn::Update3dNN, sneat.population.NeuralNetworks[0], false);
            update3d.detach();
        }
    }

    return 0;
}
