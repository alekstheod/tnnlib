#pragma once

#include <functional>
#include <numeric>
#include <utility>
#include <algorithm>
#include <cmath>

namespace nn {

    template< typename VarType >
    class SoftmaxFunction {
      public:
        typedef VarType Var;
        template< typename V >
        using use = SoftmaxFunction< V >;

      public:
        template< typename Iterator >
        Var calculate(const Var& sum, Iterator begin, Iterator end) const {
            return std::exp(sum) / std::accumulate(begin,
                                                   end,
                                                   Var{},
                                                   [](Var init, Var next) -> Var {
                                                       return init + std::exp(next);
                                                   });
        }

        template< typename Iterator >
        Var sum(Iterator begin, Iterator end, const Var& start) const {
            return std::accumulate(begin, end, start);
        }

        Var delta(const Var& output, const Var& expectedOutput) const {
            return (output - expectedOutput);
        }
    };
} // namespace nn

