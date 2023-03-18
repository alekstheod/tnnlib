#pragma once

#include <functional>
#include <numeric>
#include <utility>
#include <cmath>

namespace nn {

    template< class VarType >
    class TanhFunction {
      public:
        typedef VarType Var;

        template< typename V >
        using use = TanhFunction< V >;

      public:
        // 2 / (1 + exp(-2 * x)) - 1
        template< typename Iterator >
        Var calculate(const Var& sum, Iterator, Iterator) const {
            return Var{2.f} / (Var{1.f} + std::exp(Var{-2.f} * sum)) - Var{1.f};
        }

        template< typename Iterator >
        Var sum(Iterator begin, Iterator end, const Var& start) const {
            return std::accumulate(begin, end, start);
        }

        /**
         * Will calculate the delta of given equation for the input value.
         * @param output the output value of the neuron.
         * @return calculated differential for given input value.
         */
        Var delta(const Var& output, const Var& expectedOutput) const {
            return (output - expectedOutput) * derivate(output);
        }

        Var derivate(const Var& output) const {
            return Var{1.f} - output * output;
        }
    };
} // namespace nn
