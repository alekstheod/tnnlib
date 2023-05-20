#pragma once

#include <functional>
#include <numeric>
#include <cmath>
#include <utility>
#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>

namespace nn {

    template< typename VarType >
    struct LogScaleSoftmaxFunction {
        typedef VarType Var;

        template< typename V >
        using use = LogScaleSoftmaxFunction< V >;

        template< typename Iterator >
        Var calculate(const Var& sum, Iterator begin, Iterator end) const {
            Var sum2{};
            while(begin != end) {
                sum2 += *begin;
                begin++;
            }

            return sum / sum2;
        }

        template< typename Iterator >
        Var sum(Iterator begin, Iterator end, const Var& start) const {
            Var max = *begin;
            Iterator i = begin;
            while(i != end) {
                if(*i > max) {
                    max = *i;
                }

                i++;
            }

            Var sum{};
            while(begin != end) {
                sum += std::exp(*begin - max);
                begin++;
            }

            sum = max + std::log(sum);
            return sum;
        }

        Var delta(const Var& output, const Var& expectedOutput) const {
            return (output - expectedOutput);
        }
    };
} // namespace nn
