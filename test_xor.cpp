#include <iostream>
#include "lib/SimpleNEAT.hpp"

int main() {
    znn::Opts.InputSize = 2;
    znn::Opts.OutputSize = 1;
    znn::Opts.ActiveFunction = znn::SteependSigmoid;
    znn::Opts.DerivativeFunction = znn::DerivativeSteependSigmoid;
    znn::Opts.IterationTimes = 0;
    znn::Opts.ThreadCount = 1;
    znn::Opts.PopulationSize = 9;
    znn::Opts.ChampionKeepSize = 2;
    znn::Opts.ChampionToNewSize = 6;
    znn::Opts.KeepWorstSize = 1;
    znn::Opts.NewSize = 1;
    znn::Opts.KeepComplexSize = 1;
    znn::Opts.Enable3dNN = false;
    znn::Opts.MutateAddNeuronRate = 0.3f;
    znn::Opts.MutateAddConnectionRate = 1.f;
    znn::Opts.PrecisionFunction = znn::AbsoluteDeviation;
    znn::Opts.FitnessThreshold = 0.95f; // 如果<=0则不启用
    znn::Opts.CheckPointPath = "/tmp/xor";

    znn::SimpleNeat sneat;
    sneat.StartNew();

    const std::vector<std::vector<float>> inputs = {{0.f, 0.f},
                                                    {1.f, 1.f},
                                                    {0.f, 1.f},
                                                    {1.f, 0.f},};

    const std::vector<std::vector<float>> wanted = {{0.f},
                                                    {0.f},
                                                    {1.f},
                                                    {1.f},};

    auto best = sneat.TrainByWanted(inputs, wanted, 0, []() { return false; });

//    std::cout << "HiddenNeuronInnovations: " << znn::HiddenNeuronInnovations.size() << " ConnectionInnovations: " << znn::ConnectionInnovations.size() << std::endl;
    std::cout << "HiddenNeuronInnovations: " << sneat.population.generation.neuralNetwork.HiddenNeuronInnovations.size() << std::endl;

    std::cout << "best: geration:" << best.Gen << " fitness " << best.Fit << " neurons " << best.NN.Neurons.size() << " connections " << best.NN.Connections.size() << std::endl;

    std::cout << "neurons:\n";
    for (auto &n : best.NN.Neurons) {
        std::cout << n.Id << " " << n.Bias << std::endl;
    }

    std::cout << "connections:\n";
    for (auto &c : best.NN.Connections) {
        std::cout << c.ConnectedNeuronId[0] << " " << c.ConnectedNeuronId[1] << " " << c.Weight << std::endl;
    }

    std::cout << "predict: \n";
    for (int i = 0; i < inputs.size(); ++i) {
        std::cout << inputs[i][0] << " " << inputs[i][1] << " [" << wanted[i][0] << "] " << sneat.population.generation.neuralNetwork.FeedForwardPredict(&best.NN, inputs[i], false)[0] << std::endl;
    }

    sneat.population.generation.neuralNetwork.ExportNNToDot(best.NN, "/tmp/xor");
//    sneat.population.generation.neuralNetwork.ExportNN(best.NN, "/tmp/xor");

//    znn::Update3dNN(best.NN, true);
//
//    char ccc;
//    std::cin >> ccc;

    return 0;
}
