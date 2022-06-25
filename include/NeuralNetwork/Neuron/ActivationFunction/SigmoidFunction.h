#pragma once

#include <NeuralNetwork/Neuron/ActivationFunction/IActivationFunction.h>

#include <vector>
#include <functional>
#include <numeric>
#include <cmath>
#include <utility>

namespace nn {

    /**
     * Sigmoid function implementation.
     * Used by not-linear neural networks.
     */
    template< class VarType >
    class SigmoidFunction {
      public:
        typedef VarType Var;
        template< typename V >
        using use = SigmoidFunction< V >;

        /**
         * Empty constructor.
         */
        SigmoidFunction() : m_slope(1.0f) {
        }

        /**
         * Will calculate the equation
         * for the given input value.
         * @return the calculation result.
         */
        template< typename Iterator >
        Var calculate(const Var& sum, Iterator begin, Iterator end) const {
            Var tmp(m_slope * sum);
            Var tmp2 = tmp * Var{-1.0f};
            return Var{1.f} /
                   (Var{1.f} + std::exp(tmp2));
        }

        template< typename Iterator >
        Var sum(Iterator begin, Iterator end, const Var& start) const {
            return std::accumulate(begin, end, start);
        }

        /**
         * Will calculate the delta d=(yo-yw)*f'(s)
         * for current equation.
         * @param output the output value of the neuron.
         * @return result of calculation.
         */
        Var delta(const Var& output, const Var& expectedOutput) const {
            return (output - expectedOutput) * derivate(output);
        }

        Var derivate(const Var& output) const {
            return m_slope * output * (Var{1.0f} - output);
        }

      private:
        /**
         * Slope value.
         */
        Var m_slope;
    };
} // namespace nn
