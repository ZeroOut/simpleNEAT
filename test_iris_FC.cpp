#include <iostream>
#include "lib/SimpleNEAT.hpp"

int main() {
    znn::Opts.InputSize = 4;
    znn::Opts.OutputSize = 3;
    znn::Opts.ActiveFunction = znn::Sigmoid;
    znn::Opts.DerivativeFunction = znn::DerivativeSigmoid;
    znn::Opts.FCNN_hideLayers = {32, 32};
    znn::Opts.FitnessThreshold = 0.999f;
    znn::Opts.LearnRate = 1.f;
    znn::Opts.Update3dIntercalMs = 100;
    znn::Opts.Enable3dRandPos = false;
    znn::Opts.X_Interval3d = 1.5f;

    znn::SimpleNeat sneat;
    auto NN = sneat.population.generation.neuralNetwork.NewFCNN();

    std::vector<std::vector<float>> inputs = {{5.1f, 3.5f, 1.4f, 0.2f},
                                              {4.9f, 3.0f, 1.4f, 0.2f},
                                              {4.7f, 3.2f, 1.3f, 0.2f},
                                              {4.6f, 3.1f, 1.5f, 0.2f},
                                              {5.0f, 3.6f, 1.4f, 0.2f},
                                              {5.4f, 3.9f, 1.7f, 0.4f},
                                              {4.6f, 3.4f, 1.4f, 0.3f},
                                              {5.0f, 3.4f, 1.5f, 0.2f},
                                              {4.4f, 2.9f, 1.4f, 0.2f},
                                              {4.9f, 3.1f, 1.5f, 0.1f},
                                              {5.4f, 3.7f, 1.5f, 0.2f},
                                              {4.8f, 3.4f, 1.6f, 0.2f},
                                              {4.8f, 3.0f, 1.4f, 0.1f},
                                              {4.3f, 3.0f, 1.1f, 0.1f},
                                              {5.8f, 4.0f, 1.2f, 0.2f},
                                              {5.7f, 4.4f, 1.5f, 0.4f},
                                              {5.4f, 3.9f, 1.3f, 0.4f},
                                              {5.1f, 3.5f, 1.4f, 0.3f},
                                              {5.7f, 3.8f, 1.7f, 0.3f},
                                              {5.1f, 3.8f, 1.5f, 0.3f},
                                              {5.4f, 3.4f, 1.7f, 0.2f},
                                              {5.1f, 3.7f, 1.5f, 0.4f},
                                              {4.6f, 3.6f, 1.0f, 0.2f},
                                              {5.1f, 3.3f, 1.7f, 0.5f},
                                              {4.8f, 3.4f, 1.9f, 0.2f},
                                              {5.0f, 3.0f, 1.6f, 0.2f},
                                              {5.0f, 3.4f, 1.6f, 0.4f},
                                              {5.2f, 3.5f, 1.5f, 0.2f},
                                              {5.2f, 3.4f, 1.4f, 0.2f},
                                              {4.7f, 3.2f, 1.6f, 0.2f},
                                              {4.8f, 3.1f, 1.6f, 0.2f},
                                              {5.4f, 3.4f, 1.5f, 0.4f},
                                              {5.2f, 4.1f, 1.5f, 0.1f},
                                              {5.5f, 4.2f, 1.4f, 0.2f},
                                              {4.9f, 3.1f, 1.5f, 0.1f},
                                              {5.0f, 3.2f, 1.2f, 0.2f},
                                              {5.5f, 3.5f, 1.3f, 0.2f},
                                              {4.9f, 3.1f, 1.5f, 0.1f},
                                              {4.4f, 3.0f, 1.3f, 0.2f},
                                              {5.1f, 3.4f, 1.5f, 0.2f},
                                              {5.0f, 3.5f, 1.3f, 0.3f},
                                              {4.5f, 2.3f, 1.3f, 0.3f},
                                              {4.4f, 3.2f, 1.3f, 0.2f},
                                              {5.0f, 3.5f, 1.6f, 0.6f},
                                              {5.1f, 3.8f, 1.9f, 0.4f},
                                              {4.8f, 3.0f, 1.4f, 0.3f},
                                              {5.1f, 3.8f, 1.6f, 0.2f},
                                              {4.6f, 3.2f, 1.4f, 0.2f},
                                              {5.3f, 3.7f, 1.5f, 0.2f},
                                              {5.0f, 3.3f, 1.4f, 0.2f},
                                              {7.0f, 3.2f, 4.7f, 1.4f},
                                              {6.4f, 3.2f, 4.5f, 1.5f},
                                              {6.9f, 3.1f, 4.9f, 1.5f},
                                              {5.5f, 2.3f, 4.0f, 1.3f},
                                              {6.5f, 2.8f, 4.6f, 1.5f},
                                              {5.7f, 2.8f, 4.5f, 1.3f},
                                              {6.3f, 3.3f, 4.7f, 1.6f},
                                              {4.9f, 2.4f, 3.3f, 1.0f},
                                              {6.6f, 2.9f, 4.6f, 1.3f},
                                              {5.2f, 2.7f, 3.9f, 1.4f},
                                              {5.0f, 2.0f, 3.5f, 1.0f},
                                              {5.9f, 3.0f, 4.2f, 1.5f},
                                              {6.0f, 2.2f, 4.0f, 1.0f},
                                              {6.1f, 2.9f, 4.7f, 1.4f},
                                              {5.6f, 2.9f, 3.6f, 1.3f},
                                              {6.7f, 3.1f, 4.4f, 1.4f},
                                              {5.6f, 3.0f, 4.5f, 1.5f},
                                              {5.8f, 2.7f, 4.1f, 1.0f},
                                              {6.2f, 2.2f, 4.5f, 1.5f},
                                              {5.6f, 2.5f, 3.9f, 1.1f},
                                              {5.9f, 3.2f, 4.8f, 1.8f},
                                              {6.1f, 2.8f, 4.0f, 1.3f},
                                              {6.3f, 2.5f, 4.9f, 1.5f},
                                              {6.1f, 2.8f, 4.7f, 1.2f},
                                              {6.4f, 2.9f, 4.3f, 1.3f},
                                              {6.6f, 3.0f, 4.4f, 1.4f},
                                              {6.8f, 2.8f, 4.8f, 1.4f},
                                              {6.7f, 3.0f, 5.0f, 1.7f},
                                              {6.0f, 2.9f, 4.5f, 1.5f},
                                              {5.7f, 2.6f, 3.5f, 1.0f},
                                              {5.5f, 2.4f, 3.8f, 1.1f},
                                              {5.5f, 2.4f, 3.7f, 1.0f},
                                              {5.8f, 2.7f, 3.9f, 1.2f},
                                              {6.0f, 2.7f, 5.1f, 1.6f},
                                              {5.4f, 3.0f, 4.5f, 1.5f},
                                              {6.0f, 3.4f, 4.5f, 1.6f},
                                              {6.7f, 3.1f, 4.7f, 1.5f},
                                              {6.3f, 2.3f, 4.4f, 1.3f},
                                              {5.6f, 3.0f, 4.1f, 1.3f},
                                              {5.5f, 2.5f, 4.0f, 1.3f},
                                              {5.5f, 2.6f, 4.4f, 1.2f},
                                              {6.1f, 3.0f, 4.6f, 1.4f},
                                              {5.8f, 2.6f, 4.0f, 1.2f},
                                              {5.0f, 2.3f, 3.3f, 1.0f},
                                              {5.6f, 2.7f, 4.2f, 1.3f},
                                              {5.7f, 3.0f, 4.2f, 1.2f},
                                              {5.7f, 2.9f, 4.2f, 1.3f},
                                              {6.2f, 2.9f, 4.3f, 1.3f},
                                              {5.1f, 2.5f, 3.0f, 1.1f},
                                              {5.7f, 2.8f, 4.1f, 1.3f},
                                              {6.3f, 3.3f, 6.0f, 2.5f},
                                              {5.8f, 2.7f, 5.1f, 1.9f},
                                              {7.1f, 3.0f, 5.9f, 2.1f},
                                              {6.3f, 2.9f, 5.6f, 1.8f},
                                              {6.5f, 3.0f, 5.8f, 2.2f},
                                              {7.6f, 3.0f, 6.6f, 2.1f},
                                              {4.9f, 2.5f, 4.5f, 1.7f},
                                              {7.3f, 2.9f, 6.3f, 1.8f},
                                              {6.7f, 2.5f, 5.8f, 1.8f},
                                              {7.2f, 3.6f, 6.1f, 2.5f},
                                              {6.5f, 3.2f, 5.1f, 2.0f},
                                              {6.4f, 2.7f, 5.3f, 1.9f},
                                              {6.8f, 3.0f, 5.5f, 2.1f},
                                              {5.7f, 2.5f, 5.0f, 2.0f},
                                              {5.8f, 2.8f, 5.1f, 2.4f},
                                              {6.4f, 3.2f, 5.3f, 2.3f},
                                              {6.5f, 3.0f, 5.5f, 1.8f},
                                              {7.7f, 3.8f, 6.7f, 2.2f},
                                              {7.7f, 2.6f, 6.9f, 2.3f},
                                              {6.0f, 2.2f, 5.0f, 1.5f},
                                              {6.9f, 3.2f, 5.7f, 2.3f},
                                              {5.6f, 2.8f, 4.9f, 2.0f},
                                              {7.7f, 2.8f, 6.7f, 2.0f},
                                              {6.3f, 2.7f, 4.9f, 1.8f},
                                              {6.7f, 3.3f, 5.7f, 2.1f},
                                              {7.2f, 3.2f, 6.0f, 1.8f},
                                              {6.2f, 2.8f, 4.8f, 1.8f},
                                              {6.1f, 3.0f, 4.9f, 1.8f},
                                              {6.4f, 2.8f, 5.6f, 2.1f},
                                              {7.2f, 3.0f, 5.8f, 1.6f},
                                              {7.4f, 2.8f, 6.1f, 1.9f},
                                              {7.9f, 3.8f, 6.4f, 2.0f},
                                              {6.4f, 2.8f, 5.6f, 2.2f},
                                              {6.3f, 2.8f, 5.1f, 1.5f},
                                              {6.1f, 2.6f, 5.6f, 1.4f},
                                              {7.7f, 3.0f, 6.1f, 2.3f},
                                              {6.3f, 3.4f, 5.6f, 2.4f},
                                              {6.4f, 3.1f, 5.5f, 1.8f},
                                              {6.0f, 3.0f, 4.8f, 1.8f},
                                              {6.9f, 3.1f, 5.4f, 2.1f},
                                              {6.7f, 3.1f, 5.6f, 2.4f},
                                              {6.9f, 3.1f, 5.1f, 2.3f},
                                              {5.8f, 2.7f, 5.1f, 1.9f},
                                              {6.8f, 3.2f, 5.9f, 2.3f},
                                              {6.7f, 3.3f, 5.7f, 2.5f},
                                              {6.7f, 3.0f, 5.2f, 2.3f},
                                              {6.3f, 2.5f, 5.0f, 1.9f},
                                              {6.5f, 3.0f, 5.2f, 2.0f},
                                              {6.2f, 3.4f, 5.4f, 2.3f},
                                              {5.9f, 3.0f, 5.1f, 1.8f},};

    // Iris-setosa {1.f,0.f,0.f}
    // Iris-versicolor {0.f,1.f,0.f}
    // Iris-virginica {0.f,0.f,1.f}

    std::vector<std::vector<float>> wanted = {{1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {1.f, 0.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 1.f, 0.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},
                                              {0.f, 0.f, 1.f},};

    //    std::thread show3d([]() {
    //        znn::Show3dNN();
    //    });
    //    show3d.detach();

    float fitness = 0.f;
    while (fitness < znn::Opts.FitnessThreshold) {
        fitness = 0.f;
        ++NN.Age;
        for (int i = 0; i < inputs.size(); ++i) {
            std::vector<float> thisOutputs = sneat.population.generation.neuralNetwork.BackPropagation(&NN, inputs[i], wanted[i], false);
            fitness += znn::GetPrecision(thisOutputs, wanted[i]);
        }
        fitness /= float(inputs.size());
        if (NN.Age % 100 == 0) {
            std::cout << NN.Age << " fitness: " << fitness << "\n";
            //            znn::Update3dNN_Background(NN, false);
        }
    }

    std::cout << "predict: \n";
    for (int i = 0; i < inputs.size(); ++i) {
        auto predict = sneat.population.generation.neuralNetwork.FeedForwardPredict(&NN, inputs[i], false);
        std::cout << inputs[i][0] << " " << inputs[i][1] << inputs[i][2] << " " << inputs[i][3] << " [" << wanted[i][0] << " " << wanted[i][1] << " " << wanted[i][2] << "] " << predict[0] << " " << predict[1] << " " << predict[2] << std::endl;
    }

    std::cout << NN.Age << " fitness: " << fitness << "\n";

    return 0;
}