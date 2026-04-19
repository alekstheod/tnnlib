#pragma once

#include <cmath>
#include <limits>
#include <array>

namespace nn::bp {

    template< typename Var, size_t Size >
    struct IOptimizer {
        virtual Var operator()(size_t index, Var weight, Var gradient) = 0;

      protected:
        ~IOptimizer() = default;
    };

    // Adam optimizer: maintains moving averages of gradient and squared gradient
    template< typename Var, size_t Size >
    struct AdamOptimizer final : public IOptimizer< Var, Size > {
        Var learningRate;
        Var beta1; // exponential decay rate for first moment estimates
        Var beta2; // exponential decay rate for second moment estimates
        Var epsilon; // small value to prevent division by zero
        size_t t; // timestep

        std::array<Var, Size> m; // first moment vector
        std::array<Var, Size> v; // second moment vector

        AdamOptimizer() : learningRate(0.001f), beta1(0.9f), beta2(0.999f), epsilon(1e-8f), t(0), m{}, v{} {
        }

        AdamOptimizer(Var lr, Var b1, Var b2, Var eps = 1e-8f)
         : learningRate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0), m{}, v{} {
        }

        Var operator()(size_t index, Var weight, Var gradient) final override {
            t++;

            Var clippedGradient = gradient;
            if (std::abs(gradient) > 10.0f) {
                clippedGradient = (gradient > 0 ? 10.0f : -10.0f);
            }

            m[index] = beta1 * m[index] + (1 - beta1) * clippedGradient;
            v[index] = beta2 * v[index] + (1 - beta2) * (clippedGradient * clippedGradient);

            Var m_hat = m[index] / (1 - std::pow(beta1, t));
            Var v_hat = v[index] / (1 - std::pow(beta2, t));

            return weight - learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    };

    // Adagrad optimizer: adapts learning rate based on historical sum of squares
    template< typename Var, size_t Size >
    struct AdagradOptimizer final : public IOptimizer< Var, Size > {
        Var learningRate;
        Var epsilon;
        std::array<Var, Size> cache; // sum of squares of past gradients

        AdagradOptimizer() : learningRate(0.01f), epsilon(1e-8f), cache{} {
        }

        AdagradOptimizer(Var lr, Var eps = 1e-8f)
         : learningRate(lr), epsilon(eps), cache{} {
        }

        Var operator()(size_t index, Var weight, Var gradient) final override {
            Var clippedGradient = gradient;
            if (std::abs(gradient) > 10.0f) {
                clippedGradient = (gradient > 0 ? 10.0f : -10.0f);
            }

            cache[index] += clippedGradient * clippedGradient;
            return weight - learningRate * clippedGradient / (std::sqrt(cache[index]) + epsilon);
        }
    };

    template< typename Var, size_t Size >
    struct SgdOptimizer final : public IOptimizer< Var, Size > {
        Var learningRate;

        SgdOptimizer() : learningRate(0.01f) {
        }

        SgdOptimizer(Var lr) : learningRate(lr) {
        }

        Var operator()(size_t index, Var weight, Var gradient) final override {
            Var clippedGradient = gradient;
            if (std::abs(gradient) > 10.0f) {
                clippedGradient = (gradient > 0 ? 10.0f : -10.0f);
            }
            return weight - learningRate * clippedGradient;
        }
    };

    // Momentum optimizer
    template< typename Var, size_t Size >
    struct MomentumOptimizer final : public IOptimizer< Var, Size > {
        Var learningRate;
        Var beta; // momentum coefficient
        std::array<Var, Size> velocity; // velocity term

        MomentumOptimizer() : learningRate(0.01f), beta(0.9f), velocity{} {
        }

        MomentumOptimizer(Var lr, Var b)
         : learningRate(lr), beta(b), velocity{} {
        }

        Var operator()(size_t index, Var weight, Var gradient) final override {
            Var clippedGradient = gradient;
            if (std::abs(gradient) > 10.0f) {
                clippedGradient = (gradient > 0 ? 10.0f : -10.0f);
            }
            velocity[index] = beta * velocity[index] + (1 - beta) * clippedGradient;
            return weight - learningRate * velocity[index];
        }
    };

} // namespace nn::bp
