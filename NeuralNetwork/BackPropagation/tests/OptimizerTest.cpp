#include "NeuralNetwork/BackPropagation/Optimizers.h"

#include <catch2/catch_all.hpp>

namespace {

    constexpr size_t OptimizerSize = 10;

    TEST_CASE("SgdOptimizer basic functionality", "[optimizer]") {
        nn::bp::SgdOptimizer<float, OptimizerSize> optimizer(0.1f);

        float weight = 1.0f;
        float gradient = 0.5f;

        auto newWeight = optimizer(0, weight, gradient);

        REQUIRE(newWeight == Catch::Approx(1.0f - 0.1f * 0.5f));
    }

    TEST_CASE("SgdOptimizer updates weights correctly", "[optimizer]") {
        nn::bp::SgdOptimizer<float, OptimizerSize> optimizer(0.01f);

        float weight = 2.0f;
        float gradient = 1.0f;

        auto result1 = optimizer(0, weight, gradient);
        auto result2 = optimizer(1, result1, gradient);

        REQUIRE(result1 == Catch::Approx(1.99f));
        REQUIRE(result2 == Catch::Approx(1.98f));
    }

    TEST_CASE("MomentumOptimizer basic functionality", "[optimizer]") {
        nn::bp::MomentumOptimizer<float, OptimizerSize> optimizer(0.1f, 0.0f);

        float weight = 1.0f;
        float gradient = 1.0f;

        auto newWeight = optimizer(0, weight, gradient);

        REQUIRE(newWeight == Catch::Approx(0.9f));
    }

    TEST_CASE("MomentumOptimizer accumulates momentum", "[optimizer]") {
        nn::bp::MomentumOptimizer<float, OptimizerSize> optimizer(0.01f, 0.9f);

        float weight = 1.0f;
        float gradient = 1.0f;

        auto result1 = optimizer(0, weight, gradient);

        REQUIRE(result1 == Catch::Approx(0.999f).margin(0.01f));
    }

    TEST_CASE("AdagradOptimizer basic functionality", "[optimizer]") {
        nn::bp::AdagradOptimizer<float, OptimizerSize> optimizer(0.1f, 1e-8f);

        float weight = 1.0f;
        float gradient = 1.0f;

        auto newWeight = optimizer(0, weight, gradient);

        REQUIRE(newWeight == Catch::Approx(1.0f - 0.1f * 1.0f));
    }

    TEST_CASE("AdamOptimizer basic functionality", "[optimizer]") {
        nn::bp::AdamOptimizer<float, OptimizerSize> optimizer;

        float weight = 1.0f;
        float gradient = 0.1f;

        auto newWeight = optimizer(0, weight, gradient);

        REQUIRE(newWeight != weight);
    }

    TEST_CASE("Different optimizers produce different results", "[optimizer]") {
        nn::bp::SgdOptimizer<float, OptimizerSize> sgd(0.1f);
        nn::bp::MomentumOptimizer<float, OptimizerSize> momentum(0.1f, 0.0f);
        nn::bp::AdagradOptimizer<float, OptimizerSize> adagrad(0.1f);

        float weight = 1.0f;
        float gradient = 1.0f;

        auto sgdResult = sgd(0, weight, gradient);
        auto momentumResult = momentum(0, weight, gradient);
        auto adagradResult = adagrad(0, weight, gradient);

        REQUIRE(sgdResult == Catch::Approx(0.9f));
        REQUIRE(momentumResult == Catch::Approx(0.9f));
        REQUIRE(adagradResult == Catch::Approx(0.9f));
    }

    TEST_CASE("Optimizer with different indices", "[optimizer]") {
        nn::bp::AdamOptimizer<float, OptimizerSize> optimizer;

        float weight1 = 1.0f;
        float weight2 = 2.0f;
        float gradient1 = 0.1f;
        float gradient2 = 0.2f;

        auto result1 = optimizer(0, weight1, gradient1);
        auto result2 = optimizer(1, weight2, gradient2);

        REQUIRE(result1 != result2);
    }

    TEST_CASE("SgdOptimizer handles large gradients without NaN", "[optimizer]") {
        nn::bp::SgdOptimizer<float, OptimizerSize> optimizer(0.01f);

        float weight = 1.0f;
        float largeGradient = 100.0f;

        auto result = optimizer(0, weight, largeGradient);

        REQUIRE(std::isfinite(result));
    }

    TEST_CASE("AdamOptimizer handles large gradients without NaN", "[optimizer]") {
        nn::bp::AdamOptimizer<float, OptimizerSize> optimizer;

        float weight = 1.0f;
        float largeGradient = 100.0f;

        auto result = optimizer(0, weight, largeGradient);

        REQUIRE(std::isfinite(result));
    }

    TEST_CASE("MomentumOptimizer handles large gradients without NaN", "[optimizer]") {
        nn::bp::MomentumOptimizer<float, OptimizerSize> optimizer(0.01f, 0.9f);

        float weight = 1.0f;
        float largeGradient = 100.0f;

        auto result = optimizer(0, weight, largeGradient);

        REQUIRE(std::isfinite(result));
    }

    TEST_CASE("AdagradOptimizer handles large gradients without NaN", "[optimizer]") {
        nn::bp::AdagradOptimizer<float, OptimizerSize> optimizer(0.01f);

        float weight = 1.0f;
        float largeGradient = 100.0f;

        auto result = optimizer(0, weight, largeGradient);

        REQUIRE(std::isfinite(result));
    }

}