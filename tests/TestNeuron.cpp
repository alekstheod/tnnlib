#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>

#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include "catch.hpp"

#include <vector>

namespace {
    SCENARIO("Neuron output calculation", "[neuron][forward]") {
        GIVEN("Neuron with 3 inputs and sigmoid activation function") {
            nn::Neuron< nn::SigmoidFunction, float, 3 > neuron;
            std::vector< float > dotProducts;
            WHEN("Weights are set to 1 and bias is equal to 1") {
                neuron.setBias(1);
                for(auto i : ranges::views::ints(0, 3)) {
                    neuron.setWeight(i, 1);
                    neuron.setInput(i, 1);
                }

                THEN("The output of the neuron is 1/(1+exp(-4)) -> 0.982") {
                    const auto output =
                     neuron.calculateOutput(dotProducts.begin(), dotProducts.end());
                    REQUIRE(output == Approx(0.982f).margin(0.001));
                }
            }
            WHEN("Weights are set to 0 and bias is equal to 0") {
                neuron.setBias(0);
                for(auto i : ranges::views::ints(0, 3)) {
                    neuron.setWeight(i, 0);
                    neuron.setInput(i, 1);
                }
                THEN("The output of the neuron is 1/1+exp(0) -> 0.5") {
                    const auto output =
                     neuron.calculateOutput(dotProducts.begin(), dotProducts.end());
                    REQUIRE(output == Approx(0.5f).margin(0.001));
                }
            }
            WHEN(
             "Weights are set to 1 bias is equal to 0 and inputs are set to "
             "0") {
                neuron.setBias(0);
                for(auto i : ranges::views::ints(0, 3)) {
                    neuron.setWeight(i, 1);
                    neuron.setInput(i, 0);
                }
                THEN("The output of the neuron is 1/1+exp(0) -> 0.5") {
                    const auto output =
                     neuron.calculateOutput(dotProducts.begin(), dotProducts.end());
                    REQUIRE(output == Approx(0.5f).margin(0.001));
                }
            }
        }
    }
} // namespace
