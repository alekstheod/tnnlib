#include "NeuralNetwork/ActivationFunction/Binary.h"
#include <cstdlib>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch_all.hpp>

namespace {
    template< std::size_t result, std::size_t delta_result, std::size_t derivative_result, std::size_t sum_result >
    struct ActivationFunctionStub {
        using Var = float;
        float to_result(std::size_t value) const {
            return static_cast< float >(value) / 100.f;
        }

        template< typename Iterator >
        float calculate(float, Iterator, Iterator) const {
            return to_result(result);
        }

        float delta(float, float) const {
            return to_result(delta_result);
        }

        float derivate(float) const {
            return to_result(derivative_result);
        }

        template< typename Iterator >
        float sum(Iterator, Iterator, float) const {
            return to_result(sum_result);
        }
    };

    SCENARIO("Binary filter for activation function test",
             "[activation_function][filter]") {
        GIVEN("the activation function and a binary filter") {
            using ActivationFunction = ActivationFunctionStub< 69U, 20U, 30U, 40U >;
            WHEN("Activation function output is above threshold") {
                using Function = ActivationFunctionStub< 70U, 20U, 30U, 40U >;
                nn::Binary< Function, 69U > binary;
                THEN("binary filter outputs 1") {
                    REQUIRE(binary.calculate(0.F, 0, 1) == Catch::Approx(1.f));
                }
            }
            WHEN("Activation function output is equal to threshold") {
                using Function = ActivationFunctionStub< 69U, 20U, 30U, 40U >;
                nn::Binary< Function, 69U > binary;
                THEN("binary filter outputs 0") {
                    REQUIRE(binary.calculate(0.F, 0, 1) == Catch::Approx(0.f));
                }
            }
            WHEN("Activation function output is below to threshold") {
                using Function = ActivationFunctionStub< 69U, 20U, 30U, 40U >;
                nn::Binary< Function, 69U > binary;
                THEN("binary filter outputs 0") {
                    REQUIRE(binary.calculate(0.F, 0, 1) == Catch::Approx(0.f));
                }
            }
            WHEN("calling delta on binary function") {
                using Function = ActivationFunctionStub< 69U, 20U, 30U, 40U >;
                nn::Binary< Function, 69U > binary;
                THEN(
                 "the result is the same as when calling delta on activation "
                 "function") {
                    REQUIRE(binary.delta(0.F, 0.F) ==
                            Catch::Approx(ActivationFunction{}.delta(0.f, 0.f)));
                }
            }
            WHEN("calling sum on binary function") {
                using Function = ActivationFunctionStub< 69U, 20U, 30U, 40U >;
                nn::Binary< Function, 69U > binary;
                THEN(
                 "the result is the same as when calling sum on activation "
                 "function") {
                    REQUIRE(binary.sum(0U, 1U, 0.1F) ==
                            Catch::Approx(ActivationFunction{}.sum(0U, 1U, 0.1F)));
                }
            }
            WHEN("calling derivative on binary function") {
                using Function = ActivationFunctionStub< 69U, 20U, 30U, 40U >;
                nn::Binary< Function, 69U > binary;
                THEN(
                 "the result is the same as when calling derivative on "
                 "activation "
                 "function") {
                    REQUIRE(binary.derivate(0.f) ==
                            Catch::Approx(ActivationFunction{}.derivate(0.f)));
                }
            }
        }
    }
} // namespace
