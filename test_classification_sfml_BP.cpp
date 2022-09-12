#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include "lib/SimpleNEAT.hpp"


int main() {
    // Window
    sf::RenderWindow window(sf::VideoMode(1024, 1024), "classification", sf::Style::Titlebar | sf::Style::Close);
//    window.setFramerateLimit(60);

    sf::Event ev{};

    std::vector<sf::RectangleShape> blocks;
    std::map<std::vector<float>, int> markedBlocks;
    std::map<sf::Uint32, std::vector<sf::Vector2<float>>> targets;

    for (int i = 0; i < 64; ++i) {
        for (int ii = 0; ii < 64; ++ii) {
            sf::RectangleShape box(sf::Vector2f(16.f, 16.f));
            box.setPosition(float(i * 16), float(ii * 16));
            box.setFillColor(sf::Color(0, 0, 0, 255));
            blocks.push_back(box);
        }
    }

    sf::Vector2<int> clickPos(1025, 1025);
    sf::Color boxColor(sf::Color::Yellow);
    bool isTrainingStart = false;

    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> wantedOutputs;

    znn::NetworkGenome nn;
    znn::SimpleNeat sneat;

    uint outputLen;

    auto startTrain = [&]() {
        outputLen = targets.size();
        znn::Opts.InputSize = 2;
        znn::Opts.OutputSize = outputLen;
        znn::Opts.ActiveFunction = znn::Sigmoid;
        znn::Opts.DerivativeFunction = znn::DerivativeSigmoid;
        //    znn::Opts.IterationTimes = 3000;
        znn::Opts.LearnRate = 0.1f;
        znn::Opts.FitnessThreshold = 0.99f;
        znn::Opts.IterationCheckPoint = 0;

        nn = sneat.population.generation.neuralNetwork.NewFCNN({9,9,9,9});
//        nn = znn::NewNN();

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
    };

    bool isSolved = false;

    uint rounds = 1;

    auto singleFromLoop = [&]() {
        using namespace znn;
        float solveCount = 0;

        for (uint i = 0; i < inputs.size(); ++i) {
            if (sneat.population.generation.BackPropagation(&nn, inputs[i], wantedOutputs[i])) {
                ++solveCount;
            }
        }

        ++rounds;

        if (solveCount / float(inputs.size()) > znn::Opts.FitnessThreshold) {
            return true;
        }

        return false;
    };

    float lastFitness = 0.f;
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
            for (int cc = 0; cc < 32; ++cc) {  // substep
                if (singleFromLoop()) {
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
                pos.x += 8.f;
                pos.y += 8.f;
                if (!markedBlocks.contains({pos.x, pos.y})) {
                    auto outputs = sneat.population.generation.neuralNetwork.FeedForwardPredict(&nn, {pos.x / 1024.f, pos.y / 1024.f});
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
                pos.x += 8.f;
                pos.y += 8.f;
                targets[boxColor.toInteger()].push_back(pos);
                markedBlocks[{pos.x, pos.y}] = 1;
                clickPos.x = 1025;
                clickPos.y = 1025;
            }
            window.draw(b);
        }

        if (isSolved) {
            isSolved = false;
        }

        window.display(); // Tell app window is done drawing

    }
    return 0;
}
