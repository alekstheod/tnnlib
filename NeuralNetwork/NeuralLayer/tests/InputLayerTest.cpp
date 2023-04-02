#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

namespace {
    SCENARIO("InputLayer basic calculation",
             "[layer][basic][sigmoid][forward]") {
        GIVEN("A neural layer with 2 neurons and 2 features (inputs)") {
            nn::InputLayer< nn::Neuron, nn::SigmoidFunction, 2 > layer;
            layer.setInput(0, {0.6f, 0.5});
            layer.setInput(1, {0.3f, 0.4f});
            WHEN("calculateOutputs is called") {
                layer.calculateOutputs();
                THEN(
                 "Neurons outputs are equal with a result of the function 1 / "
                 "(1 + std::exp(-dot_pruduct))") {
                    REQUIRE(layer[0].getOutput() == Approx(1.f / (1.f + std::exp(-1.1f))));
                    REQUIRE(layer[1].getOutput() == Approx(1.f / (1.f + std::exp(-0.7f))));
                }
            }
        }
    }
} // namespace
