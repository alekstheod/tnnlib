#pragma once

#include <cmath>
#include <limits>

namespace nn::bp {

    template< typename Var >
    struct IOptimizer {
        virtual Var operator()(Var weight, Var gradient) = 0;

      protected:
        ~IOptimizer() = default;
    };

    // Adam optimizer: maintains moving averages of gradient and squared gradient
    template< typename Var >
    struct AdamOptimizer final : public IOptimizer< Var > {
        Var learningRate;
        Var beta1; // exponential decay rate for first moment estimates
        Var beta2; // exponential decay rate for second moment estimates
        Var epsilon; // small value to prevent division by zero
        size_t t; // timestep

        // State variables (these would normally be per-parameter)
        Var m; // first moment vector
        Var v; // second moment vector

        AdamOptimizer(Var lr = 0.001, Var b1 = 0.9, Var b2 = 0.999, Var eps = 1e-8)
         : learningRate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0), m(0), v(0) {
        }

        Var operator()(Var weight, Var gradient) final override {
            t++;
            m = beta1 * m + (1 - beta1) * gradient;
            v = beta2 * v + (1 - beta2) * (gradient * gradient);

            // Bias correction
            Var m_hat = m / (1 - std::pow(beta1, t));
            Var v_hat = v / (1 - std::pow(beta2, t));

            return weight - learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    };

    // Adagrad optimizer: adapts learning rate based on historical sum of squares
    template< typename Var >
    struct AdagradOptimizer final : public IOptimizer< Var > {
        Var learningRate;
        Var epsilon;
        Var cache; // sum of squares of past gradients

        AdagradOptimizer(Var lr = 0.01, Var eps = 1e-8)
         : learningRate(lr), epsilon(eps), cache(0) {
        }

        Var operator()(Var weight, Var gradient) final override {
            cache += gradient * gradient;
            return weight - learningRate * gradient / (std::sqrt(cache) + epsilon);
        }
    };

    template< typename Var >
    struct SgdOptimizer final : public IOptimizer< Var > {
        Var learningRate;

        SgdOptimizer(Var lr) : learningRate(lr) {
        }

        Var operator()(Var weight, Var gradient) final override {
            return weight - learningRate * gradient;
        }
    };

    // Momentum optimizer
    template< typename Var >
    struct MomentumOptimizer final : public IOptimizer< Var > {
        Var learningRate;
        Var beta; // momentum coefficient
        Var velocity; // velocity term

        MomentumOptimizer(Var lr, Var b = 0.9)
         : learningRate(lr), beta(b), velocity(0) {
        }

        Var operator()(Var weight, Var gradient) final override {
            velocity = beta * velocity + (1 - beta) * gradient;
            return weight - learningRate * velocity;
        }
    };

} // namespace nn::bp
