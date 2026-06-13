#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch_all.hpp>

namespace {
    SCENARIO("InputLayer basic calculation",
             "[layer][basic][sigmoid][forward]") {
        GIVEN("A neural layer with 2 neurons and 2 features (inputs)") {
            nn::InputLayer< nn::Neuron, nn::SigmoidFunction, 2, 2 > layer;
            layer[0][0].weight = 1.f;
            layer[0][0].value = 0.6f;
            layer[0][1].weight = 1.f;
            layer[0][1].value = 0.5f;
            layer[0].setBias(0.f);
            layer[1][0].weight = 1.f;
            layer[1][0].value = 0.3f;
            layer[1][1].weight = 1.f;
            layer[1][1].value = 0.4f;
            layer[1].setBias(0.f);
            WHEN("calculateOutputs is called") {
                using Context = std::tuple<std::array<float, 2>>;
                Context ctx;
                layer.calculateOutputs<Context, 0>(ctx);
                THEN(
                 "Neurons outputs are equal with a result of the function 1 / "
                 "(1 + std::exp(-dot_pruduct))") {
                    REQUIRE(layer[0].getOutput() ==
                            Catch::Approx(1.f / (1.f + std::exp(-1.1f))));
                    REQUIRE(layer[1].getOutput() ==
                            Catch::Approx(1.f / (1.f + std::exp(-0.7f))));
                }
            }
        }
    }
} // namespace
