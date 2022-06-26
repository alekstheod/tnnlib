#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <cstdlib>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

namespace {
    SCENARIO("sigmoid activation function test",
             "[activation_function][sigmoid][calculate]") {
        GIVEN("sigmoid activation function") {
            nn::SigmoidFunction< float > sigmoid;
            WHEN("provided dot product is 0") {
                THEN("the calculation result is near 1/2") {
                    REQUIRE(sigmoid.calculate(0.f, 0, 2) == Approx(0.5f));
                }
                THEN("the derivate result is near 0.25") {
                    REQUIRE(sigmoid.derivate(0.5f) == Approx(0.25f));
                }
                THEN(
                 "the delta is near 0.25 * of difference between expected and "
                 "actual outputs") {
                    REQUIRE(sigmoid.delta(0.5f, 1.f) == Approx((0.5f - 1.f) * 0.25f));
                }
            }
            WHEN("provided dot product is very high") {
                THEN("the calculation result is near 1") {
                    REQUIRE(sigmoid.calculate(100.f, 0, 2) == Approx(1.f));
                }
                THEN("the derivate result is near 0") {
                    REQUIRE(sigmoid.derivate(1.f) == Approx(0.f));
                }
                THEN("the delta is near 0") {
                    REQUIRE(sigmoid.delta(1.f, 0.f) == Approx(0.f));
                }
            }
            WHEN("provided dot product is very low") {
                THEN("the calculation result is near 0") {
                    REQUIRE(sigmoid.calculate(-100.f, 0, 2) == Approx(0.f));
                }
                THEN("the derivate result is near 0") {
                    REQUIRE(sigmoid.derivate(0.f) == Approx(0.f));
                }
                THEN("the delta is near 0") {
                    REQUIRE(sigmoid.delta(0.f, 1.f) == Approx(0.f));
                }
            }
            WHEN("provided dot product is increasing") {
                float pred_result{-1.f};
                for(float i = -100; i < 100; i++) {
                    THEN("the calculation result is increasing") {
                        auto current_result = sigmoid.calculate(i, 0, 2);
                        REQUIRE(current_result > pred_result);
                        pred_result = current_result;
                    }
                }
            }
        }
    }
} // namespace
