#include <iostream>
#include "lib/MyNEAT.hpp"

int main() {
    znn::Opts.InputSize = 2;
    znn::Opts.OutputSize = 1;
    znn::Opts.ActiveFunction = znn::SteependSigmoid;
    znn::Opts.IterationTimes = 0;
    znn::Opts.FitnessThreshold = 0.98f;
    znn::Opts.IterationCheckPoint = 0;
    znn::Opts.ThreadCount = 8;
//    znn::Opts.MutateAddNeuronRate = 0.45f;
//    znn::Opts.MutateAddConnectionRate = 0.99f;
//    znn::Opts.NewSize = 0;
//    znn::Opts.KeepLastSize = 0;
//    znn::Opts.ChampionToNewSize = 50;

    znn::StartNew();

    const std::vector<std::vector<float>> inputs = {
            {0.f, 0.f},
            {1.f, 1.f},
            {0.f, 1.f},
            {1.f, 0.f},

    };

    const std::vector<std::vector<float>> wanted = {
            {0.f},
            {0.f},
            {1.f},
            {1.f},
    };

    auto best = znn::TrainByWanted(inputs, wanted);

//    std::cout << "HiddenNeuronInnovations: " << znn::HiddenNeuronInnovations.size() << " ConnectionInnovations: " << znn::ConnectionInnovations.size() << std::endl;
    std::cout << "HiddenNeuronInnovations: " << znn::HiddenNeuronInnovations.size() << std::endl;

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
        std::cout << inputs[i][0] << " " << inputs[i][1] << " [" << wanted[i][0] << "] " << znn::FeedForwardPredict(&best.NN, inputs[i])[0] << std::endl;
    }

//    znn::ExportNetwork(best.NN, "/tmp/xor00");
//    znn::ExportInnovations("/tmp/xor00");
//    znn::ExportNetworkToDot(best.NN, "/tmp/xxx00");

    return 0;
}
