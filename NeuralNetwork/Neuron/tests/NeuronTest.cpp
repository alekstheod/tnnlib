#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/Neuron/RecurrentNeuron.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"

#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

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
                    const auto output = neuron.calculateOutput(neuron.calcDotProduct(),
                                                               dotProducts.begin(),
                                                               dotProducts.end());
                    REQUIRE(output == Catch::Approx(0.982f).margin(0.001));
                }
            }
            WHEN("Weights are set to 0 and bias is equal to 0") {
                neuron.setBias(0);
                for(auto i : ranges::views::ints(0, 3)) {
                    neuron.setWeight(i, 0);
                    neuron.setInput(i, 1);
                }
                THEN("The output of the neuron is 1/1+exp(0) -> 0.5") {
                    const auto output = neuron.calculateOutput(neuron.calcDotProduct(),
                                                               dotProducts.begin(),
                                                               dotProducts.end());
                    REQUIRE(output == Catch::Approx(0.5f).margin(0.001));
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
                    const auto output = neuron.calculateOutput(neuron.calcDotProduct(),
                                                               dotProducts.begin(),
                                                               dotProducts.end());
                    REQUIRE(output == Catch::Approx(0.5f).margin(0.001));
                }
            }
        }
        GIVEN("Recurrent neuron with 1 input") {
            nn::RecurrentNeuron< nn::SigmoidFunction, float, 1 > unit;
            WHEN("Provided number of inputs is 1") {
                THEN("Neuron has one more hidden recurrent input") {
                    REQUIRE(unit.size() == 2);
                }
            }
            WHEN("Outputs are calculated") {
                std::array< float, 2 > vals = {0.f, 0.f};
                auto output =
                 unit.calculateOutput(0.5f, std::begin(vals), std::end(vals));
                THEN(
                 "the recurrent input is equal to the output of the neuron") {
                    REQUIRE(output == unit[unit.size() - 1].value);
                }
            }
        }
    }
} // namespace
