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

#ifndef EquationH
#define EquationH
#include <map>

namespace nn {

    /**
     * Equation interface.
     * Represent the basic interface for equations
     * implementation.
     */

    template < class FunctionType > class IActivationFunction {
        public:
        typedef typename FunctionType::Var Var;

        private:
        FunctionType m_function;

        public:
        /**
        * @brief Empty constructor.
        */
        IActivationFunction () : m_function () {
        }

        /**
         * @brief constructor with a function as argument.
         */
        IActivationFunction (FunctionType function) : m_function (function) {
        }

        /**
         * Will calculate the equation
         * for given input value.
         * @param neuronWeight the neurons weight used in bep algorithm
         * @param inputs neuron inputs.
         * @return the calculated output value.
         */
        template < typename Iterator > Var calculate (const Var& sum, Iterator begin, Iterator end) const {
            return m_function.calculate (sum, begin, end);
        }

        /**
         * Will calculate the differential of given equation for the input value.
         * @param output the output value of the neuron.
         * @return calculated differential for given input value.
         */
        Var delta (const Var& output, const Var& expectedOutput) const {
            return m_function.delta (output, expectedOutput);
        }

        /**
         * @brief will calculate the derivate of equation.
         * @param output the output - point for which the value of the derivate will be calculated.
         * @return the calculated value of the derivate.
         */
        Var derivate (const Var& output) const {
            return m_function.derivate (output);
        }

        /**
         * @brief This method will calculate the output of the neural network without the activation function,
         * @brief typically the sum of the weights*inputs
         * @param begin iterator which points to the first mult of weight and input (weight*input).
         * @param end the end of the container.
         * @param start the starting point from which the inputs will be accumulated, typically a bias for a neuron.
         * @return the accumulated weight*input + bias of the neuron.
         */
        template < typename Iterator > Var sum (Iterator begin, Iterator end, const Var& start) const {
            return m_function.sum (begin, end, start);
        }

        FunctionType* operator* () {
            return &m_function;
        }

        /**
         * Destructor.
         */
        ~IActivationFunction () {
        }
    };
}

#endif
// kate: indent-mode cstyle; replace-tabs on;
