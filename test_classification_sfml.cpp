#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include "lib/SimpleNEAT.hpp"


int main() {
    // Window
    sf::RenderWindow window(sf::VideoMode(1024, 1024), "classification", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(40);

    sf::Event ev{};


    std::map<znn::NetworkGenome *, float> populationFitness;
    std::vector<znn::NetworkGenome *> orderedPopulation;
    std::vector<znn::NetworkGenome *> orderedByComplex;

    std::vector<sf::RectangleShape> blocks;
    std::map<std::vector<float>, int> markedBlocks;
    std::map<sf::Uint32, std::vector<sf::Vector2<float>>> targets;

    for (int i = 0; i < 128; ++i) {
        for (int ii = 0; ii < 128; ++ii) {
            sf::RectangleShape box(sf::Vector2f(8.f, 8.f));
            box.setPosition(float(i * 8), float(ii * 8));
            box.setFillColor(sf::Color(0, 0, 0, 255));
            blocks.push_back(box);
        }
    }

    sf::Vector2<int> clickPos(1025, 1025);
    sf::Color boxColor(sf::Color::Yellow);
    bool isTrainingStart = false;

    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> wantedOutputs;

    uint outputLen;

    auto startTrain = [&]() {
        outputLen = targets.size();
        znn::Opts.InputSize = 2;
        znn::Opts.OutputSize = outputLen;
        znn::Opts.ActiveFunction = znn::Sigmoid;
        //    znn::Opts.IterationTimes = 3000;
        znn::Opts.FitnessThreshold = 0.999f;
        znn::Opts.IterationCheckPoint = 0;
//        znn::Opts.ThreadCount = 16;
//        znn::Opts.MutateAddNeuronRate = 0.5f;
        znn::Opts.MutateAddConnectionRate = 0.99f;
        znn::Opts.PopulationSize = 300;
        znn::Opts.ChampionKeepSize = 30;
        znn::Opts.NewSize = 10;
        znn::Opts.KeepWorstSize = 0;
        znn::Opts.ChampionToNewSize = 90;
        znn::Opts.KeepComplexSize = 30;
        znn::Opts.WeightRange = 16.f;
        znn::Opts.BiasRange = 8.f;
        znn::Opts.MutateWeightDirectOrNear = 0.36f;
        znn::Opts.MutateWeightNearRange = 6;
        znn::Opts.MutateBiasDirectOrNear = 0.36f;

        znn::StartNew();

        if (outputLen < 2 || outputLen > 4) {
            std::cerr << "Error: outputLen " << outputLen << "\n";
            exit(0);
        }

        std::vector<std::vector<float>> targetVec(outputLen);

        switch (outputLen) {
            case 2:
                targetVec[0] = {0.f, 1.f};
                targetVec[1] = {1.f, 0.f};
                break;
            case 3:
                targetVec[0] = {0.f, 0.f, 1.f};
                targetVec[1] = {0.f, 1.f, 0.f};
                targetVec[2] = {1.f, 0.f, 0.f};
                break;
            case 4:
                targetVec[0] = {0.f, 0.f, 0.f, 1.f};
                targetVec[1] = {0.f, 0.f, 1.f, 0.f};
                targetVec[2] = {0.f, 1.f, 0.f, 0.f};
                targetVec[3] = {1.f, 0.f, 0.f, 0.f};
        }

        int colorNum = 0;
        for (auto &t : targets) {
            for (auto &p : t.second) {
                inputs.push_back(std::vector<float>{p.x / 1024.f, p.y / 1024.f});
                wantedOutputs.push_back(targetVec[colorNum]);
            }
            ++colorNum;
        }

        isTrainingStart = true;

        populationFitness = znn::CalculateFitnessByWanted(inputs, wantedOutputs);
        orderedPopulation = znn::OrderByFitness(populationFitness);
        orderedByComplex = znn::OrderByComplex();
    };

    bool isSolved = false;

    uint rounds = 1;

    auto singleFromLoop = [&]() {
        using namespace znn;

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
                    ++index;
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
//        exit(0);

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

        if (rounds % 100 == 0 || populationFitness[orderedPopulation[0]] > Opts.FitnessThreshold) {
            std::cout << "gen: " << rounds << " " << orderedPopulation[0] << " " << orderedPopulation[0]->Neurons.size() << " " << orderedPopulation[0]->Connections.size() << " fitness: "
                      << populationFitness[orderedPopulation[0]] << " Most complex: " << orderedByComplex[0]->Neurons.size() << std::endl;
        }
        ++rounds;

        struct result {
            znn::NetworkGenome nn;
            bool isOver;
        };

        if (populationFitness[orderedPopulation[0]] > Opts.FitnessThreshold) {
            auto simplifiedBestNN = SimplifyRemoveDisable(*orderedPopulation[0]);
            auto compressedLeftBestNN = SimplifyRemoveUselessConnectionLeft(simplifiedBestNN);
            auto compressedRightBestNN = SimplifyRemoveUselessConnectionRight(compressedLeftBestNN);
//            ExportNN(compressedRightBestNN, "./champion");
            ExportNNToDot(compressedRightBestNN, "./champion");
            isTrainingStart = false;
            isSolved = true;
            return result{
                    .nn = *orderedPopulation[0],
                    .isOver = true,
            };
        }

        return result{
                .nn = *orderedPopulation[0],
                .isOver = false,
        };
    };

    float lastFitness = 0.f;
    znn::NetworkGenome bestNN;
    bool isLoopDone = true;
//    BS::thread_pool thisPool(1);

    //Gmae Loop
    while (window.isOpen()) {
        while (window.pollEvent(ev)) {
            switch (ev.type) {
                case sf::Event::Closed:
                    window.close();
                    break;
                case sf::Event::KeyPressed:
                    if (ev.key.code == sf::Keyboard::Enter) {
                        std::cout << "Start training\n";
                        std::cout << "Colors: " << targets.size() << "\n";
                        startTrain(); // Train
                    } else if (ev.key.code == sf::Keyboard::Num0) {
                        boxColor = sf::Color::Yellow;
                        std::cout << "Set color to yellow\n";
                    } else if (ev.key.code == sf::Keyboard::Num1) {
                        boxColor = sf::Color::Red;
                        std::cout << "Set color to red\n";
                    } else if (ev.key.code == sf::Keyboard::Num2) {
                        boxColor = sf::Color::Green;
                        std::cout << "Set color to green\n";
                    } else if (ev.key.code == sf::Keyboard::Num3) {
                        boxColor = sf::Color::Blue;
                        std::cout << "Set color to blue\n";
                    }
                    break;
                case sf::Event::MouseButtonPressed:
                    if (ev.mouseButton.button == sf::Mouse::Left) {
                        clickPos = sf::Mouse::getPosition(window);
                        std::cout << clickPos.x << ", " << clickPos.y << "\n";
                    }
                    break;
            }
        }

        // Update
        if (isLoopDone && isTrainingStart) {
            isLoopDone = false;
//            thisPool.submit([&]() {
            for (int cc = 0; cc < 32; ++cc) {
                auto best = singleFromLoop();
                bestNN = best.nn;
                if (best.isOver) {
                    break;
                }
            }
            isLoopDone = true;
//            });
        }

        // Render
        window.clear(sf::Color(0, 0, 0, 255)); // Clear old frame

        // Draw game
        for (auto &b : blocks) {
            if (isTrainingStart || isSolved) {
                auto pos = b.getPosition();
                pos.x += 4.f;
                pos.y += 4.f;
                if (!markedBlocks.contains({pos.x, pos.y})) {
                    auto outputs = znn::FeedForwardPredict(&bestNN, {pos.x / 1024.f, pos.y / 1024.f});
                    switch (outputLen) {
                        case 2: {
//                            std::cout << outputs[0] << " " << outputs[1] << "\n";
                            auto thisValue = int(outputs[0] / (outputs[0] + outputs[1]) * 255);
                            b.setFillColor(sf::Color(thisValue, thisValue, thisValue, 200));
                        }
                            break;
                        case 3:
                            if (outputs[0] > outputs[1] && outputs[0] > outputs[2]) {
                                b.setFillColor(sf::Color(int(outputs[0] * 255), 0, 0, 200));
                            } else if (outputs[1] > outputs[0] && outputs[1] > outputs[2]) {
                                b.setFillColor(sf::Color(0, int(outputs[1] * 255), 0, 200));
                            } else {
                                b.setFillColor(sf::Color(0, 0, int(outputs[2] * 255), 200));
                            }
                            break;
                        case 4:
                            if (outputs[0] > outputs[1] && outputs[0] > outputs[2] && outputs[0] > outputs[3]) {
                                b.setFillColor(sf::Color(int(outputs[0] * 255), 0, 0, 200));
                            } else if (outputs[1] > outputs[0] && outputs[1] > outputs[2] && outputs[1] > outputs[3]) {
                                b.setFillColor(sf::Color(0, int(outputs[1] * 255), 0, 200));
                            } else if (outputs[2] > outputs[0] && outputs[2] > outputs[1] && outputs[2] > outputs[3]) {
                                b.setFillColor(sf::Color(0, 0, int(outputs[2] * 255), 200));
                            } else {
                                auto thisValue = int(outputs[3] * 255);
                                b.setFillColor(sf::Color(thisValue, thisValue, thisValue, 200));
                            }
                    }
                }
            }

            if (!isTrainingStart && b.getGlobalBounds().contains(float(clickPos.x), float(clickPos.y))) {
                b.setFillColor(boxColor);
                auto pos = b.getPosition();
                pos.x += 4.f;
                pos.y += 4.f;
                targets[boxColor.toInteger()].push_back(pos);
                markedBlocks[{pos.x, pos.y}] = 1;
                clickPos.x = 1025;
                clickPos.y = 1025;
            }
            window.draw(b);
        }

        if (isSolved) {
//            window.setFramerateLimit(1);
            isSolved = false;
        }

        window.display(); // Tell app window is done drawing

    }
    return 0;
}
