#include "lib/SimpleNEAT.hpp"
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

const uint scale = 30;

int main() {
    // load neural network
    znn::Opts.InputSize = 784;
    znn::Opts.OutputSize = 10;
    znn::Opts.ActiveFunction = znn::Sigmoid;
    znn::Opts.Enable3dNN = true;
    znn::Opts.FCNN_hideLayers = {25};
    znn::Opts.X_Interval3d = 5.f;
    znn::Opts.Update3dIntercalMs = 100;
    znn::Opts.WeightRange = 1.f;
    znn::Opts.BiasRange = 3.f;

    znn::SimpleNeat sneat;
    auto nn = sneat.StartDeploy("MNIST");

    sf::RenderWindow window(sf::VideoMode(28 * scale, 28 * scale), "MNIST", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(120);

    sf::Event ev{};
    srandom((unsigned) clock());

    bool isKeepLeft = false;
    bool isStart = false;
    bool isAutoTest = false;

    std::vector<sf::RectangleShape> blocks;
    std::map<sf::RectangleShape *, uint> markedBlocks;
    std::vector<float> inputData(784, 0.f);
    std::vector<float> inputEmptyData(784, 0.f);
    std::vector<std::vector<float>> trainData;

    for (int i = 0; i < 28; ++i) {
        for (int ii = 0; ii < 28; ++ii) {
            sf::RectangleShape box(sf::Vector2f(scale * 1.f, scale * 1.f));
            box.setPosition(float(ii * scale), float(i * scale));
            box.setFillColor(sf::Color(0, 0, 0, 255));
            blocks.push_back(box);
        }
    }

    std::string fileName = "../MNIST_train.csv";
    std::ifstream input_file(fileName);

    if (input_file.is_open()) {
        trainData = znn::ImportCSV(fileName, false);  // auto test
    }

    auto lastTime = std::chrono::system_clock::now();

    //Gmae Loop
    while (window.isOpen()) {
        while (window.pollEvent(ev)) {
            switch (ev.type) {
                case sf::Event::Closed:
                    window.close();
                    break;
                case sf::Event::KeyPressed:
                    if (ev.key.code == sf::Keyboard::Space) {
                        if (isAutoTest) {
                            isAutoTest = false;
                            isStart = false;
                            window.setFramerateLimit(120);
                            znn::Opts.Update3dIntercalMs = 100;
                            markedBlocks.clear();
                            inputData = inputEmptyData;
                            sneat.population.generation.neuralNetwork.FCNNFeedForwardPredict(&nn, inputEmptyData, true);
                            break;
                        }
                        if (input_file.is_open()) {
                            isAutoTest = true;
                            isStart = true;
                            window.setFramerateLimit(1);
                            znn::Opts.Update3dIntercalMs = 900;
                            break;
                        }
                    }
                case sf::Event::MouseButtonPressed:
                    if (ev.mouseButton.button == sf::Mouse::Left) {
                        isKeepLeft = true;
                    }
                    if (ev.mouseButton.button == sf::Mouse::Right && !isAutoTest) {
                        markedBlocks.clear();
                        sneat.population.generation.neuralNetwork.FCNNFeedForwardPredict(&nn, inputEmptyData, true);
                        inputData = inputEmptyData;
                        isStart = false;
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

        if (isKeepLeft && !isAutoTest) {
            if (!isStart) {
                isStart = true;
            }
            auto thisPos = sf::Mouse::getPosition(window);
            int index = thisPos.y / scale * 28 + thisPos.x / scale;
            blocks[index].setFillColor(sf::Color::White);
            markedBlocks[&blocks[index]] = 1;
            inputData[index] = 1.f;
//            std::cout << index << "\n";
        }

        if (isAutoTest) {
            for (uint b = 0; b < blocks.size(); ++b) {  // auto test
                blocks[b].setFillColor(sf::Color::Black);
            }
            auto chooseData = trainData[random() % trainData.size()];  // auto test
            int number = int(chooseData[0]);  // auto test
            std::printf("%d - ", number);

            std::vector<float> thisInput;
            bool isHarder = random() % 100 < 50;

            for (uint ii = 0; ii < inputData.size(); ++ii) {
                if (isHarder && chooseData[ii + 1] < 191.25f) {
                    thisInput.push_back(0.f);
                } else {
                    thisInput.push_back(chooseData[ii + 1] / 255.f);
                }
            }

            inputData = thisInput;  // auto test

            for (uint b = 0; b < blocks.size(); ++b) {  // auto test
                if (inputData[b] > 0.f) {
                    blocks[b].setFillColor(sf::Color(int(inputData[b] * 255.f), int(inputData[b] * 255.f), int(inputData[b] * 255.f), 255));
                    //                    blocks[b].setFillColor(sf::Color(255, 255, 255, 255));
                    markedBlocks[&blocks[b]] = 1;
                }
            }
        }

        auto nowTime = std::chrono::system_clock::now();
        if ((isStart && (nowTime - lastTime).count() > 300000000 && isKeepLeft) || isAutoTest) {
            lastTime = nowTime;
            auto predict = sneat.population.generation.neuralNetwork.FCNNFeedForwardPredict(&nn, inputData, false);
            for (auto &si : znn::SortIndexes(predict)) {
                std::printf("%i[%.2f], ", si, predict[si]);
            }
            std::printf("\n");
        }

        window.clear(sf::Color(0, 0, 0, 255)); // Clear old frame

        // Draw game
        for (auto &mb : markedBlocks) {
            window.draw(*mb.first);
        }

        window.display(); // Tell app window is done drawing

        if (isAutoTest) {
            markedBlocks.clear();
        }
    }
    return 0;
}