#pragma once

namespace nn {

    /**
     * Equation interface.
     * Represent the basic interface for equations
     * implementation.
     */

    template< class FunctionType >
    class IActivationFunction {
      public:
        typedef typename FunctionType::Var Var;

      private:
        FunctionType m_function;

      public:
        /**
         * @brief Empty constructor.
         */
        IActivationFunction() : m_function() {
        }

        /**
         * @brief constructor with a function as argument.
         */
        IActivationFunction(FunctionType function) : m_function(function) {
        }

        /**
         * Will calculate the equation
         * for given input value.
         * @param neuronWeight the neurons weight used in bep algorithm
         * @param inputs neuron inputs.
         * @return the calculated output value.
         */
        template< typename Iterator >
        Var calculate(const Var& sum, Iterator begin, Iterator end) const {
            return m_function.calculate(sum, begin, end);
        }

        /**
         * Will calculate the differential of given equation for the input
         * value.
         * @param output the output value of the neuron.
         * @return calculated differential for given input value.
         */
        Var delta(const Var& output, const Var& expectedOutput) const {
            return m_function.delta(output, expectedOutput);
        }

        /**
         * @brief will calculate the derivate of equation.
         * @param output the output - point for which the value of the derivate
         * will be calculated.
         * @return the calculated value of the derivate.
         */
        Var derivate(const Var& output) const {
            return m_function.derivate(output);
        }

        /**
         * @brief This method will calculate the output of the neural network
         * without the activation function,
         * @brief typically the sum of the weights*inputs
         * @param begin iterator which points to the first mult of weight and
         * input (weight*input).
         * @param end the end of the container.
         * @param start the starting point from which the inputs will be
         * accumulated, typically a bias for a neuron.
         * @return the accumulated weight*input + bias of the neuron.
         */
        template< typename Iterator >
        Var sum(Iterator begin, Iterator end, const Var& start) const {
            return m_function.sum(begin, end, start);
        }

        FunctionType* operator*() {
            return &m_function;
        }
    };
} // namespace nn

