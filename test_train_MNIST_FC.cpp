#include <iostream>
#include "lib/SimpleNEAT.hpp"

int main() {
    znn::Opts.InputSize = 784;
    znn::Opts.OutputSize = 10;
    znn::Opts.ActiveFunction = znn::Sigmoid;
    znn::Opts.DerivativeFunction = znn::DerivativeSigmoid;
    znn::Opts.ThreadCount = 16;
    znn::Opts.FCNN_hideLayers = {30};
    znn::Opts.FitnessThreshold = 0.999f;
    znn::Opts.LearnRate = 1.f;
    znn::Opts.Update3dIntercalMs = 1000;
    znn::Opts.Enable3dRandPos = false;
    znn::Opts.Enable3dNN = false;
    znn::Opts.WeightRange = 1.f;
    znn::Opts.BiasRange = 3.f;

    znn::SimpleNeat sneat;
    znn::NetworkGenome NN = sneat.population.generation.neuralNetwork.NewFCNN();
//    znn::NetworkGenome NN = sneat.population.generation.neuralNetwork.ImportNN("MNIST");

    if (znn::Opts.Enable3dNN) {
        std::thread show3d([]() {
            znn::Show3dNN();
        });
        show3d.detach();
    }

    auto trainData = znn::ImportCSV("../MNIST_train.csv", false);  // https://github.com/sbussmann/kaggle-mnist
    std::cout << "size: " << trainData.size() << "\n";

    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> wanted;

    for (auto &d : trainData) {
        std::vector<float> thisInput;

        for (uint i = 0; i < 784; ++i) {
            if (d[i + 1] > 0.f) {
                thisInput.push_back(d[i + 1] / 255.f);
            } else {
                thisInput.push_back(0.f);
            }
        }

        inputs.push_back(thisInput);

        std::vector<float> thisWanted = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        thisWanted[uint(d[0])] = 1.f;
        wanted.push_back(thisWanted);
    }

    int rounds = 0;
    float fitness = 0.9f;

    while (fitness < znn::Opts.FitnessThreshold) {
        ++rounds;
        fitness = 0.f;

        for (int i = 0; i < inputs.size(); ++i) {
            std::vector<float> thisOutputs = sneat.population.generation.neuralNetwork.BackPropagation(&NN, inputs[i], wanted[i], true);
            fitness += znn::GetPrecision(thisOutputs, wanted[i]);

            if (i % 1000 == 0) {
                sneat.population.generation.neuralNetwork.ExportNN(NN, "MNIST");
                sneat.population.generation.neuralNetwork.ExportInnovations("MNIST");
                std::cout << "r: " << rounds << " i: " << i << " " << fitness / float(i + 1) << "\n";

                if (znn::Opts.Enable3dNN) {
                    znn::tPool.push_task(znn::Update3dNN, NN, false);
                }
            }
        }
        fitness /= float(inputs.size());

//        sneat.population.generation.neuralNetwork.ExportNN(NN, "MNIST");
//        sneat.population.generation.neuralNetwork.ExportInnovations("MNIST");

        std::cout << "r: " << rounds << " " << fitness << "\n----------------------------------------\n";
//        break;
    }

    std::cout << "finished rounds: " << rounds << " " << fitness << "\n";
    return 0;
}
