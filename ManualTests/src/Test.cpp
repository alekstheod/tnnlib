#include <NeuralNetwork/LearningAlgorithm/BackPropagation/BepAlgorithm.h>
#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/ActivationFunction/SoftmaxFunction.h>
#include <NeuralNetwork/ActivationFunction/TanhFunction.h>
#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/Perceptron/ComplexLayer.h>
#include <NeuralNetwork/Perceptron/Perceptron.h>
#include <NeuralNetwork/SOM/K2DNeighbourhood.h>
#include <NeuralNetwork/SOM/K2DPosition.h>
#include <NeuralNetwork/SOM/KNode.h>
#include <NeuralNetwork/SOM/KohonenMap.h>

#include <Design/Factory.h>
#include <Design/Singleton.h>
#include <System/Time.h>

#include <cereal/archives/xml.hpp>
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
                     nn::NeuralLayer< nn::Neuron, nn::TanhFunction, 2 >,
                     nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 5 >,
                     nn::ComplexNeuralLayer< nn::Neuron< nn::TanhFunction >, nn::Neuron< nn::SigmoidFunction > >,
                     nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 1 > >;

    typedef BepAlgorithm< Perceptron > Algo;
    Algo algorithm(0.09f);
    std::array< Algo::Prototype, 4 > prototypes = {Algo::Prototype{{0.f, 1.f}, {1.f}},
                                                   Algo::Prototype{{1.f, 0.f}, {1.f}},
                                                   Algo::Prototype{{1.f, 1.f}, {0.f}},
                                                   Algo::Prototype{{0.f, 0.f}, {0.f}}};

    unsigned int numOfEpochs =
     argc < 2 ? std::numeric_limits< unsigned int >::max() : std::atoi(argv[1]);

    Perceptron perceptron =
     algorithm.calculate(prototypes.begin(),
                         prototypes.end(),
                         [numOfEpochs](unsigned int epoch, float error) {
                             std::cout << "Epoch: " << epoch
                                       << " error: " << error << std::endl;
                             return error > 0.01f && epoch < numOfEpochs;
                         });

    using Memento = Perceptron::Memento;
    Memento memento = perceptron.getMemento();
    Perceptron perceptron2;
    {
        std::stringstream strm;
        {
            cereal::XMLOutputArchive oa(strm);
            oa << cereal::make_nvp("perceptron", memento);
        }

        Memento memento2;


        std::string str = strm.str();
        std::stringstream strStream(str);
        cereal::XMLInputArchive ia(strStream);
        ia >> memento2;
        perceptron2.setMemento(memento);
    }

    std::array< float, 2 > outputs{0};
    std::array< float, 2 > input1{0, 0};
    perceptron2.calculate(input1.begin(), input1.end(), outputs.begin());
    std::cout << "0 0 " << outputs[0] << std::endl;

    std::array< float, 2 > input2{1, 0};
    perceptron2.calculate(input2.begin(), input2.end(), outputs.begin());
    std::cout << "1 0 " << outputs[0] << std::endl;

    std::array< float, 2 > input3{1, 1};
    perceptron2.calculate(input3.begin(), input3.end(), outputs.begin());
    std::cout << "1 1 " << outputs[0] << std::endl;

    std::array< float, 2 > input4{0, 1};
    perceptron2.calculate(input4.begin(), input4.end(), outputs.begin());
    std::cout << "0 1 " << outputs[0] << std::endl;

    /// Kohonen map implementation
    typedef kohonen::K2DPosition< float, 5 > Position;
    typedef kohonen::K2DNeighbourhood< kohonen::KNode< Position > > Neighbourhood;
    typedef nn::kohonen::KohonenMap< Neighbourhood, 25, 3 > KohonenMap;

    KohonenMap kohMap;

    typedef KohonenMap::InputType InputType;
    std::vector< InputType > inputsData;
    inputsData.push_back({{0.f, 255.f, 0.f}});
    inputsData.push_back({{255.f, 0.f, 0.f}});
    inputsData.push_back({{0.f, 0.f, 255.f}});

    kohMap.calculateWeights(inputsData.begin(), inputsData.end(), 10000, 0.4f, 8.0f);
    return 0;
}
