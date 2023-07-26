#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include "lib/SimpleNEAT.hpp"

int main() {
    // Window
    sf::RenderWindow window(sf::VideoMode(1024, 1024), "classification", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(30);

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

    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> wantedOutputs;

    bool isTrainingStart = false;

    uint outputLen;

    znn::SimpleNeat sneat;
    znn::BestOne bestOne{.Fit = 0.f};
    znn::NetworkGenome choosingNN;

    bool isInitialed = false;

    auto startTrain = [&]() {
        isTrainingStart = true;

        outputLen = targets.size();

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
        for (auto &t: targets) {
            for (auto &p: t.second) {
                inputs.push_back(std::vector<float>{p.x / 1024.f, p.y / 1024.f});
                wantedOutputs.push_back(targetVec[colorNum]);
            }
            ++colorNum;
        }

        if (!isInitialed) {
            isInitialed = true;

            znn::Opts.InputSize = 2;
            znn::Opts.OutputSize = outputLen;
            znn::Opts.ActiveFunction = znn::Sigmoid;
            znn::Opts.DerivativeFunction = znn::DerivativeSigmoid;
            znn::Opts.IterationTimes = 0;
            znn::Opts.FitnessThreshold = 0.999f;
            znn::Opts.IterationCheckPoint = 0;
            znn::Opts.ThreadCount = 16;
            znn::Opts.MutateAddNeuronRate = 0.3f;
            znn::Opts.MutateAddConnectionRate = 0.99f;
            znn::Opts.PopulationSize = 64;
            znn::Opts.ChampionKeepSize = 8;
            znn::Opts.NewSize = 1;
            znn::Opts.KeepWorstSize = 0;
            znn::Opts.ChampionToNewSize = 24;
            znn::Opts.KeepComplexSize = 0;
            znn::Opts.WeightRange = 3.f;
            znn::Opts.BiasRange = 3.f;
            znn::Opts.MutateWeightDirectOrNear = 0.36f;
            znn::Opts.MutateWeightNearRange = 6;
            znn::Opts.MutateBiasDirectOrNear = 0.36f;
            znn::Opts.LearnRate = 0.3f;
            znn::Opts.Enable3dNN = true;
//            znn::Opts.StartWithFCNN = true;
//            znn::Opts.FCNN_hideLayers = {8, 8};
            znn::Opts.Update3dIntercalMs = 100;
            znn::Opts.X_Interval3d = 0.3f;
            znn::Opts.Zy_Interval3d = 1.5f;

            sneat.StartNew();
        }

        std::thread start([&]() {
            bestOne = sneat.TrainByWanted(inputs, wantedOutputs, 0, [&isTrainingStart]() { return !isTrainingStart; });
            isTrainingStart = false;
        });
        start.detach();
    };

    std::string fpsText;
    int fpsCounter = 0;

    std::thread fpsCount([&fpsCounter, &fpsText]() {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            fpsText = std::to_string(fpsCounter);
            fpsCounter = 0;
        }
    });
    fpsCount.detach();

    //Gmae Loop
    while (window.isOpen()) {
        while (window.pollEvent(ev)) {
            switch (ev.type) {
                case sf::Event::Closed:
                    window.close();
                    break;
                case sf::Event::KeyPressed:
                    if (ev.key.code == sf::Keyboard::Space) {
                        std::cout << "Start training\n";
                        std::cout << "Colors: " << targets.size() << "\n";
                        if (!isTrainingStart) {
                            startTrain(); // Train
                        } else {
                            isTrainingStart = false;
                        }
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
                    if (ev.mouseButton.button == sf::Mouse::Left && !isTrainingStart) {
                        clickPos = sf::Mouse::getPosition(window);
//                        std::cout << clickPos.x << ", " << clickPos.y << "\n";
                    }
                    break;
            }
        }

        // Update
        znn::mtx.lock();
        if (isTrainingStart) {
            choosingNN = sneat.population.NeuralNetworks[0];
        } else if (bestOne.Fit > 0.f) {
            choosingNN = bestOne.NN;
        }
        znn::mtx.unlock();


        // Render
//        window.clear(sf::Color(0, 0, 0, 255)); // Clear old frame

        // Draw game
        for (auto &b: blocks) {
            auto pos = b.getPosition();
            pos.x += 8.f;
            pos.y += 8.f;
            if (isTrainingStart && !markedBlocks.contains({pos.x, pos.y})) {
                auto outputs = sneat.population.generation.neuralNetwork.FeedForwardPredict(&choosingNN, {pos.x / 1024.f, pos.y / 1024.f}, false);
                switch (outputLen) {
                    case 2: {
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

            if (b.getGlobalBounds().contains(float(clickPos.x), float(clickPos.y))) {
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
        window.display(); // Tell app window is done drawing

        ++fpsCounter;
        window.setTitle("fps: " + fpsText);
    }

    return 0;
}
