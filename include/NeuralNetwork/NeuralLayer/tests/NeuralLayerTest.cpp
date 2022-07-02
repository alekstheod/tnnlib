#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

namespace {
    SCENARIO("NeuralLayer basic calculation",
             "[layer][basic][sigmoid][forward]") {
        GIVEN(
         "A neural layer with 2 neurons with 2 inputs set to 1 with a weight "
         "1") {
            nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 2 > layer;
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
                 "Neurons outputs are equal with a result of the function 1 / "
                 "(1 + std::exp(-dot_pruduct))") {
                    REQUIRE(layer[0].getOutput() == Approx(1.f / (1.f + std::exp(-3.f))));
                    REQUIRE(layer[1].getOutput() == Approx(1.f / (1.f + std::exp(-3.f))));
                }
            }
        }
    }
} // namespace
