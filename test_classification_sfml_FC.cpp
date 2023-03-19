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
    znn::NetworkGenome NN;

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
            znn::Opts.FCNN_hideLayers = {18, 24, 12};
            znn::Opts.FitnessThreshold = 0.999f;
            znn::Opts.LearnRate = 0.3f;

            NN = sneat.population.generation.neuralNetwork.NewFCNN();

            std::thread show3d([](){
                znn::Show3dNN();
            });
            show3d.detach();
        }

        std::thread start([&]() {
            int rounds = 0;
            float fitness = 0.f;
            while (isTrainingStart) {
                ++rounds;
                fitness = 0.f;
                for (int i = 0; i < inputs.size(); ++i) {
                    std::vector<float> thisOutputs = sneat.population.generation.BackPropagation(&NN, inputs[i], wantedOutputs[i]);
                    fitness += znn::GetPrecision(thisOutputs, wantedOutputs[i]);
                }
                fitness /= float(inputs.size());
                if (fitness >= znn::Opts.FitnessThreshold) {
                    break;
                }
                if (rounds % 100 == 0) {
                    std::cout << rounds << " fitness: " << fitness << "\n";
                    znn::tPool.push_task(znn::Update3dNN, NN, false);
                }
            }
            std::cout << rounds << " fitness: " << fitness << "\n";
            isTrainingStart = false;
        });

        start.detach();
    };

    std::string fpsText;
    int fpsCounter = 0;

    std::thread fpsCount([&fpsCounter, &fpsText](){
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

        // Render
        window.clear(sf::Color(0, 0, 0, 255)); // Clear old frame

        // Draw game
        for (auto &b: blocks) {
            auto pos = b.getPosition();
            pos.x += 8.f;
            pos.y += 8.f;
            if (isTrainingStart && !markedBlocks.contains({pos.x, pos.y})) {
                auto outputs = sneat.population.generation.neuralNetwork.FeedForwardPredict(&NN, {pos.x / 1024.f, pos.y / 1024.f}, false);
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
        window.setTitle("fps: "+ fpsText);
    }

    return 0;
}
