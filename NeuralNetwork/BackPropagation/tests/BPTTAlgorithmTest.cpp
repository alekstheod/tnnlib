#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/BackPropagation/BpttAlgorithm.h"
#include "NeuralNetwork/Neuron/RecurrentNeuron.h"
#include "NeuralNetwork/Perceptron/Perceptron.h"

#include <catch2/catch.hpp>

namespace {

    using RnnLayer = nn::NeuralLayer< nn::RecurrentNeuron, nn::TanhFunction, 2, 2 >;
    using Perceptron =
     nn::Perceptron< float, nn::InputLayer< nn::Neuron, nn::TanhFunction, 2, 1 >, RnnLayer >;
    using Algo = nn::bp::BpttAlgorithm< Perceptron >;
    using Prototype = typename nn::bp::BpttAlgorithm< Perceptron >::Prototype;
    using Input = typename Algo::Input;

    SCENARIO("Back error propagation through time",
             "[back-propagation][variable-length-inputs][backward]") {
        GIVEN("BpttAlgorithm") {
            nn::bp::BpttAlgorithm< Perceptron > algo{0.001f};
            WHEN("Executing algorithm") {
                // clang-format off
                std::vector< Prototype > prototypes{
					 Prototype{{{Input{1.0f}}, {Input{0.f}}, {Input{0.5f}}, {Input{0.5f}}}, {1.f}},
					 Prototype{{{Input{1.0f}}, {Input{0.5f}}, {Input{0.5f}}}, {0.f}},
				};
                // clang-format on

                THEN("Perceptron is calculated") {
                    auto perceptron =
                     algo.calculate(prototypes, [](auto epoch, auto error) {
                         return epoch >= 2;
                     });
                }
            }
        }
    }
} // namespace
