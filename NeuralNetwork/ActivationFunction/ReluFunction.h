#pragma once

#include <numeric>

namespace nn {

    template< typename VarType >
    struct ReluFunction {

        template< typename V >
        using use = ReluFunction< V >;

        using Var = VarType;

        /**
         * Will calculate the equation
         * for the given input value.
         * @return the calculation result.
         */
        template< typename Iterator >
        Var calculate(const Var& sum, Iterator begin, Iterator end) const {
            return sum >= 0 ? sum : 0;
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
            return output > 0 ? 1 : 0;
        }
    };
} // namespace nn
