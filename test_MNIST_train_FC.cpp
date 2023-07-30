#include <iostream>
#include "lib/SimpleNEAT.hpp"

int main() {
    znn::Opts.InputSize = 784;
    znn::Opts.OutputSize = 10;
    znn::Opts.ActiveFunction = znn::Sigmoid;
    znn::Opts.DerivativeFunction = znn::DerivativeSigmoid;
    znn::Opts.PrecisionFunction = znn::AbsoluteDeviation;
    znn::Opts.ThreadCount = 16;
    znn::Opts.FCNN_hideLayers = {25};
    znn::Opts.FitnessThreshold = 0.99f;
    znn::Opts.LearnRate = 1.f;
    znn::Opts.FFCNNInsteadOfFCNN = true;
    znn::Opts.WeightRange = 1.f;
    znn::Opts.BiasRange = 3.f;
    znn::Opts.NewNNWeightRange = 1.f;
    znn::Opts.NewNNBiasRange = 3.f;

    int batchSize = 1000;

    znn::SimpleNeat sneat;
    znn::NetworkGenome NN = sneat.population.generation.neuralNetwork.NewFCNN();
//    znn::NetworkGenome NN = sneat.population.generation.neuralNetwork.ImportNN("MNIST");

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
    float fitness = 0.f;

    while (fitness < znn::Opts.FitnessThreshold) {
        ++rounds;
        fitness = 0.f;

        for (int i = 0; i < batchSize; ++i) {
            int choseingIndex = random() % inputs.size();
            std::vector<float> prepairedInput = inputs[choseingIndex];

            if (random() % 100 < 50) {
                for (uint ii = 0; ii < inputs[choseingIndex].size(); ++ii) {
                    if (inputs[choseingIndex][ii] < 0.75f) {
                        prepairedInput[ii] = 0.f;
                    }
                }
            }

            std::vector<float> thisOutputs = sneat.population.generation.neuralNetwork.BackPropagation(&NN, prepairedInput, wanted[choseingIndex], true);
            fitness += znn::GetPrecision(thisOutputs, wanted[choseingIndex]);
        }

        fitness /= float(batchSize);

        std::cout << "r: " << rounds << " f: " << fitness << "\n";

        sneat.population.generation.neuralNetwork.ExportNN(NN, "MNIST");
        sneat.population.generation.neuralNetwork.ExportInnovations("MNIST");
    }

    std::cout << "finished rounds: " << rounds << " " << fitness << "\n";
    return 0;
}
