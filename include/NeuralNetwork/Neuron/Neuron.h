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

#ifndef NeuronH
#define NeuronH

#include <NeuralNetwork/Neuron/INeuron.h>
#include <NeuralNetwork/Neuron/Input.h>
#include <NeuralNetwork/Serialization/NeuronMemento.h>
#include <NeuralNetwork/Neuron/ActivationFunction/IActivationFunction.h>

#include <System/Time.h>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <map>
#include <utility>

namespace nn {
    /**
     * Neuron class.
     * Represent the neuron in the Neural layer.
     * Contains equation for output value calculation
     * and list of accepted inputs.
     */
    namespace detail {
        template< typename OutputFunctionType, std::size_t inputsNumber >
        class Neuron {
          public:
            using OutputFunction = IActivationFunction< OutputFunctionType >;
            using Var = typename OutputFunction::Var;
            using Memento = NeuronMemento< Var, inputsNumber >;
            using Input = nn::Input< Var >;

            /// @brief a list of the inputs first is the weight, second is the
            /// value
            using Inputs = typename Memento::Inputs;
            typedef typename Inputs::const_iterator iterator;

            template< typename VarType >
            using use =
             Neuron< typename OutputFunctionType::template use< VarType >, inputsNumber >;

            template< std::size_t inputs >
            using adjust = Neuron< OutputFunctionType, inputs >;

            /**
             * Initialization constructor.
             * @param inputsNumber the number of inputs for current neuron.
             * @exception NNException thrown on object initialization failure.
             */
            Neuron()
             : m_bias(utils::createRandom< Var >(1)),
               m_output(boost::numeric_cast< Var >(0)),
               m_sum(boost::numeric_cast< Var >(0)) {
                static_assert(inputsNumber > 0, "Invalid number of inputs");
            }

            /// @brief see @ref INeuron
            iterator begin() const {
                return m_inputs.begin();
            }

            /// @brief see @ref INeuron
            iterator end() const {
                return m_inputs.end();
            }

            /// @brief see @ref INeuron
            std::size_t size() const {
                return m_inputs.size();
            }

            /// @brief see @ref INeuron
            Input& operator[](std::size_t id) {
                return m_inputs[id];
            }

            const Input& operator[](const ::size_t id) const {
                return m_inputs[id];
            }

            /// @brief see @ref INeuron
            void setWeight(std::size_t weightId, const Var& weight) {
                m_inputs[weightId].weight = weight;
            }

            /// @brief see @ref INeuron
            const Var& getBias() const {
                return m_bias;
            }

            /// @brief see @ref INeuron
            const Var& getWeight(std::size_t weightId) const {
                return m_inputs[weightId].weight;
            }

            /// @brief see @ref INeuron
            const Memento getMemento() const {
                return Memento{m_bias, m_inputs};
            }

            /// @brief see @ref INeuron
            void setMemento(const Memento& memento) {
                const auto& inputs = memento.inputs;
                std::copy(inputs.begin(), inputs.end(), m_inputs.begin());
                m_bias = memento.bias;
            }

            /// @brief see @ref INeuron
            void setInput(unsigned int inputId, const Var& value) {
                m_inputs[inputId].value = value;
            }

            /// @brief see @ref INeuron
            Var calcDotProduct() const {
                const auto calcInput = [](const Input& input) {
                    return input.value * input.weight;
                };

                auto begin = boost::make_transform_iterator(m_inputs.cbegin(), calcInput);
                auto end = boost::make_transform_iterator(m_inputs.cend(), calcInput);
                return m_activationFunction.sum(begin, end, m_bias);
            }

            /// @brief see @ref INeuron
            const Var& getOutput() const {
                return m_output;
            }

            /// @brief see @ref INeuron
            unsigned int getInputsNumber() const {
                return m_inputs.size();
            }

            /// @brief see @ref INeuron
            void setBias(Var weight) {
                m_bias = weight;
            }

            /// @brief see @ref INeuron
            const Var& getNeuronWeight() const {
                return m_bias;
            }

            /// @brief see @ref INeuron
            template< typename Iterator >
            const Var& calculateOutput(Iterator begin, Iterator end) {
                m_output = m_activationFunction.calculate(calcDotProduct(), begin, end);
                return m_output;
            }

            template< typename Test >
            void supportTest(Test&);

          private:
            /**
             * @brief Instance of output calculation equation.
             * @brief The equation should be provided by implementation of
             * IEquationFactory interface.
             */
            OutputFunction m_activationFunction;

            /**
             * @brief List of neurons inputs.
             */
            Inputs m_inputs;

            /**
             * @brief Neurons weight.
             * @brief Needed in order to improve the flexibility of neural
             * network.
             */
            Var m_bias;

            /**
             * @brief The neurons output.
             * @brief The output will be calculated with using the instance of
             * calculation equation.
             */
            Var m_output;

            /**
             * @brief Needed in order to calculate the neurons output value.
             */
            Var m_sum;
        };
    } // namespace detail

    template< template< class > class OutputFunctionType, typename VarType, std::size_t inputsNumber >
    using Neuron = detail::Neuron< OutputFunctionType< VarType >, inputsNumber >;
} // namespace nn

#endif
// kate: indent-mode cstyle; replace-tabs on;
