#include "NeuralNetwork//BackPropagation/BepAlgorithm.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/ActivationFunction/SoftmaxFunction.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/Perceptron/ComplexLayer.h"
#include "NeuralNetwork/Perceptron/Perceptron.h"
#include "NeuralNetwork/Serialization/Cereal.h"

#include <Design/Factory.h>
#include <Design/Singleton.h>
#include <System/Time.h>

#include <cereal/archives/json.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

#include <array>
#include <iostream>
#include <sstream>
#include <time.h>
#include <fstream>

using namespace nn;
using namespace bp;
using namespace std;
using namespace utils;

/*
 *
 */
int main(int argc, char** argv) {
    using Perceptron =
     nn::Perceptron< float,
                     nn::InputLayer< nn::Neuron, nn::TanhFunction, 2, 1 >,
                     nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 5 >,
                     nn::ComplexNeuralLayer< nn::Neuron< nn::TanhFunction >, nn::Neuron< nn::SigmoidFunction > >,
                     nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 1 > >;

    typedef BepAlgorithm< Perceptron > Algo;
    Algo algorithm(0.09f);
    using Input = Perceptron::Input;
    std::array< Algo::Prototype, 4 > prototypes = {
     Algo::Prototype{{Input{0.f}, Input{1.f}}, {1.f}},
     Algo::Prototype{{Input{1.f}, Input{0.f}}, {1.f}},
     Algo::Prototype{{Input{1.f}, Input{1.f}}, {0.f}},
     Algo::Prototype{{Input{0.f}, Input{0.f}}, {0.f}}};

    unsigned int numOfEpochs =
     argc < 2 ? std::numeric_limits< unsigned int >::max() : std::strtol(argv[1]);

    Perceptron perceptron =
     algorithm.calculate(prototypes.begin(),
                         prototypes.end(),
                         [numOfEpochs](unsigned int epoch, float error) {
                             std::cout << "Epoch: " << epoch
                                       << " error: " << error << std::endl;
                             return error > 0.001f && epoch < numOfEpochs;
                         });

    using Memento = Perceptron::Memento;
    const auto store = [](const auto& memento) {
        std::stringstream strm;
        {
            cereal::JSONOutputArchive oa(strm);
            oa << cereal::make_nvp("perceptron", memento);
            std::cout << strm.str() << std::endl;
        }
        return strm.str();
    };

    const auto restore = [](const auto& str) {
        std::stringstream strm{str};
        cereal::JSONInputArchive ia(strm);
        Memento memento;
        ia >> memento;
        return memento;
    };

    Perceptron perceptron2;
    const auto mementoStr = store(perceptron.getMemento());
    perceptron2.setMemento(restore(mementoStr));

    std::array< float, 2 > outputs{0};
    std::array< Input, 2 > input1{Input{0.f}, Input{0.f}};
    perceptron2.calculate(input1.begin(), input1.end(), outputs.begin());
    std::cout << "0 0 " << outputs[0] << std::endl;

    std::array< Input, 2 > input2{Input{1}, Input{0}};
    perceptron2.calculate(input2.begin(), input2.end(), outputs.begin());
    std::cout << "1 0 " << outputs[0] << std::endl;

    std::array< Input, 2 > input3{Input{1}, Input{1}};
    perceptron2.calculate(input3.begin(), input3.end(), outputs.begin());
    std::cout << "1 1 " << outputs[0] << std::endl;

    std::array< Input, 2 > input4{Input{0}, Input{1}};
    perceptron2.calculate(input4.begin(), input4.end(), outputs.begin());
    std::cout << "0 1 " << outputs[0] << std::endl;

    return 0;
}
