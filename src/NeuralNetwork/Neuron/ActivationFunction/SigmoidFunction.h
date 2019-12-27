/**
*  Copyright (c) 2011, Alex Theodoridis
*  All rights reserved.

*  Redistribution and use in source and binary forms, with
*  or without modification, are permitted provided that the
*  following conditions are met:
*  Redistributions of source code must retain the above
*  copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above
*  copyright notice, this list of conditions and the following
*  disclaimer in the documentation and/or other materials
*  provided with the distribution.

*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
*  AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
*  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
*  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
*  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
*  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
*  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE,
*  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

#ifndef SigmaEquationH
#define SigmaEquationH
#include <Neuron/ActivationFunction/IActivationFunction.h>

#include <vector>
#include <functional>
#include <numeric>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>

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
            Var tmp2 = tmp * boost::numeric_cast< Var >(-1.0f);
            return boost::numeric_cast< Var >(1.0f) /
                   (boost::numeric_cast< Var >(1.0f) + std::exp(tmp2));
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
            return m_slope * output * (boost::numeric_cast< Var >(1.0f) - output);
        }

      private:
        /**
         * Slope value.
         */
        Var m_slope;
    };
} // namespace nn

#endif
// kate: indent-mode cstyle; replace-tabs on;
