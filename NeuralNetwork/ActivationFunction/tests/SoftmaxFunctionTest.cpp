#include "NeuralNetwork/ActivationFunction/SoftmaxFunction.h"

#include <cstdlib>
#include <iostream>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

namespace {
    SCENARIO("softmax activation function test",
             "[activation_function][softmax][calculate]") {
        GIVEN("softmax activation function") {
            nn::SoftmaxFunction< float > softmax;
            THEN(
             "the delta is a difference between current and expected "
             "output") {
                REQUIRE(softmax.delta(4.f, 2.f) == Approx(2.f));
                REQUIRE(softmax.delta(5.f, 2.f) == Approx(3.f));
            }
            WHEN("provided dot product is 0 and others neuron outputs are 0") {
                std::vector< float > neighbors(10, 0.f);
                THEN("the calculation result is near 0.1") {
                    REQUIRE(softmax.calculate(0.f, std::begin(neighbors), std::end(neighbors)) ==
                            Approx(0.1f));
                }
            }
            WHEN("dot product increases") {
                std::vector< float > neighbors(10, 1.f);
                for(float i = -100.f; i < 100.f; i++) {
                    THEN(
                     "the calculation result is following the formula "
                     "exp(x)/sum(neighbors)") {
                        REQUIRE(
                         softmax.calculate(i, std::begin(neighbors), std::end(neighbors)) ==
                         Approx(std::exp(i) / (10.f * std::exp(1.f))).epsilon(0.001f));
                    }
                }
            }
        }
    }
} // namespace
