#include <NeuralNetwork/ActivationFunction/TanhFunction.h>
#include <cstdlib>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

namespace {
    SCENARIO("tahn activation function test",
             "[activation_function][tahn][calculate]") {
        GIVEN("tahn activation function") {
            nn::TanhFunction< float > sigmoid;
            WHEN("provided dot product is 0") {
                THEN("the calculation result is near 1") {
                    REQUIRE(sigmoid.calculate(0.f, 0, 0) == Approx(0.f));
                }
                THEN("the derivate result is near 1") {
                    REQUIRE(sigmoid.derivate(0.0f) == Approx(1.f));
                }
                THEN(
                 "the delta is near difference between expected and "
                 "actual outputs") {
                    REQUIRE(sigmoid.delta(0.0f, 1.f) == Approx((-1.f)));
                }
            }
        }
    }
} // namespace
