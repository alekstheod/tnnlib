#pragma once

#include <cmath>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

namespace nn {

    template< typename VarType >
    class BiopolarSigmoidFunction {
      public:
        typedef VarType Var;

        template< typename V >
        using use = BiopolarSigmoidFunction< V >;

      public:
        template< typename Iterator >
        Var calculate(const Var& sum, Iterator begin, Iterator end) const {
            Var tmp = sum * Var{-2.0f};
            return Var{2.0f} / (Var{1.f} + std::exp(tmp)) - Var{1.0f};
        }

        Var delta(const Var& output, const Var& expectedOutput) const {
            return (output - expectedOutput) * calculateDerivate(output);
        }

        template< typename Iterator >
        Var sum(Iterator begin, Iterator end, const Var& start) const {
            return std::accumulate(begin, end, start);
        }

        Var derivate(const Var& output) const {
            return (Var{1.0f} - output * output) / Var{2.0f};
        }
    };
} // namespace nn
