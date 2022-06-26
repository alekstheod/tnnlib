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
         * Will calculate the equation
         * for the given input value.
         * @return the calculation result.
         */
        template< typename Iterator >
        Var calculate(const Var& sum, Iterator, Iterator) const {
            return calculate(sum);
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
        Var calculate(const Var& sum) const {
            return Var{1.f} / (Var{1.f} + std::exp(-(m_slope * sum)));
        }

        /**
         * Slope value.
         */
        Var m_slope{1.f};
    };
} // namespace nn
