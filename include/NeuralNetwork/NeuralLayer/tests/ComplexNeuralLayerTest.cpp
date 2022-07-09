#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/ActivationFunction/SoftmaxFunction.h>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

namespace {
    SCENARIO("Complex neural layer basic calculation",
             "[layer][basic][sigmoid][forward]") {
        GIVEN(
         "A neural layer with 2 neurons of 2 different "
         "types") {
            nn::ComplexNeuralInputLayer< 2U, float, nn::Neuron< nn::SigmoidFunction, float >, nn::Neuron< nn::SoftmaxFunction, float > > layer;
            nn::Input< float > one{1.f, 1.f};
            layer[0][0] = one;
            layer[0][1] = one;
            layer[1][0] = one;
            layer[1][1] = one;
            layer[0].setBias(1.f);
            layer[1].setBias(1.f);
            WHEN("calculateOutputs is called") {
                layer.calculateOutputs();
                THEN(
                 "Neurons output of the SigmoidNeuron is equal to 1 / (1 + "
                 "std::exp(-dot_pruduct))") {
                    REQUIRE(layer[0].getOutput() == Approx(1.f / (1.f + std::exp(-3.f))));
                }
                THEN("Neurons output of the SoftmaxNeuron is equal to ") {
                    REQUIRE(layer[1].getOutput() ==
                            Approx(std::exp(3.f) / (std::exp(3.f) + std::exp(3.f))));
                }
            }
        }
    }
} // namespace
