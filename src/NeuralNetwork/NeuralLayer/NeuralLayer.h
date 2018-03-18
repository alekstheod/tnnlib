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

#ifndef NeuralLayerH
#define NeuralLayerH

#include <NeuralNetwork/INeuralLayer.h>
#include <NeuralNetwork/Neuron/INeuron.h>
#include <NeuralNetwork/Serialization/NeuralLayerMemento.h>

#include <boost/bind.hpp>
#include <boost/bind/placeholders.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <range/v3/all.hpp>

#include <algorithm>
#include <array>
#include <functional>

namespace nn {

    namespace detail {
        /**
         * Represent the NeuralLayer in perceptron.
         */
        template< class NeuronType, std::size_t neuronsNumber, std::size_t inputsNumber >
        class NeuralLayer {
          public:
            using Neuron = INeuron< typename NeuronType::template resize< inputsNumber > >;
            using Var = typename Neuron::Var;
            using NeuronMemento = typename Neuron::Memento;
            using Memento = NeuralLayerMemento< NeuronMemento, neuronsNumber >;

            template< template< class > class NewType >
            using wrap =
             NeuralLayer< NewType< NeuronType >, neuronsNumber, inputsNumber >;

            template< unsigned int inputs >
            using resize = NeuralLayer< NeuronType, neuronsNumber, inputs >;

            template< typename VarType >
            using use =
             NeuralLayer< typename NeuronType::template use< VarType >, neuronsNumber, inputsNumber >;
            static constexpr unsigned int CONST_NEURONS_NUMBER = neuronsNumber;
            static constexpr unsigned int CONST_INPUTS_NUMBER = inputsNumber;

          private:
            /**
             * A list of the neurons.
             */
            typedef typename std::vector< Neuron > Container;
            Container m_neurons;

          public:
            NeuralLayer() : m_neurons(neuronsNumber) {
            }

            /**
             * Constructor will initialize the layer by the given inputs number
             * and neurons number.
             */
            static_assert(neuronsNumber > 0,
                          "Invalid template argument neuronsNumber == 0");
            static_assert(inputsNumber > 0,
                          "Invalid template argument inputsNumber <= 1");

            /**
             * @see {INeuralLayer}
             */
            auto cbegin() const -> decltype(m_neurons.cbegin()) {
                return m_neurons.cbegin();
            }

            /**
             * @see {INeuralLayer}
             */
            auto cend() const -> decltype(m_neurons.cend()) {
                return m_neurons.cend();
            }

            /**
             * @see {INeuralLayer}
             */
            auto begin() -> decltype(m_neurons.begin()) {
                return m_neurons.begin();
            }

            /**
             * @see {INeuralLayer}
             */
            auto end() -> decltype(m_neurons.end()) {
                return m_neurons.end();
            }

            /**
             * @see {INeuralLayer}
             */
            auto size() const -> decltype(m_neurons.size()) {
                return m_neurons.size();
            }

            /**
             * @see {INeuralLayer}
             */
            const Neuron& operator[](unsigned int id) const {
                return m_neurons[id];
            }

            /**
             * @see {INeuralLayer}
             */
            Neuron& operator[](unsigned int id) {
                return m_neurons[id];
            }

            /**
             * @see {INeuralLayer}
             */
            void setInput(unsigned int inputId, const Var& value) {
                std::for_each(m_neurons.begin(),
                              m_neurons.end(),
                              std::bind(&Neuron::setInput, std::placeholders::_1, inputId, value));
            }

            const Var& getBias(unsigned int neuronId) const {
                return m_neurons[neuronId].getBias();
            }

            /**
             * @see {INeuralLayer}
             */
            const Var& getInputWeight(unsigned int neuronId, unsigned int weightId) const {
                return m_neurons[neuronId].getWeight(weightId);
            }

            /**
             * @see {INeuralLayer}
             */
            const Memento getMemento() const {
                using namespace ranges;
                Memento memento;
                memento.neurons =
                 m_neurons |
                 view::transform([](const Neuron& n) { return n.getMemento(); });
                return memento;
            }

            /**
             * @see {INeuralLayer}
             */
            void setMemento(const Memento& memento) {
                using namespace ranges;
                m_neurons = memento.neurons | view::transform([](const NeuronMemento& m) {
                                Neuron neuron;
                                neuron->setMemento(m);
                                return neuron;
                            });
            }

            /**
             * @see {INeuralLayer}
             */
            Var getOutput(unsigned int outputId) const {
                return m_neurons[outputId].getOutput();
            }

            /**
             * @see {INeuralLayer}
             */
            template< typename Layer >
            void calculateOutputs(Layer& nextLayer) {
                auto begin =
                 boost::make_transform_iterator(m_neurons.begin(),
                                                boost::bind(&Neuron::calcDotProduct, _1));
                auto end =
                 boost::make_transform_iterator(m_neurons.end(),
                                                boost::bind(&Neuron::calcDotProduct, _1));
                for(unsigned int i = 0; i < m_neurons.size(); i++) {
                    nextLayer.setInput(i, m_neurons[i].calculateOutput(begin, end));
                }
            }

            /**
             * @see {INeuralLayer}
             */
            void calculateOutputs() {
                auto begin =
                 boost::make_transform_iterator(m_neurons.begin(),
                                                boost::bind(&Neuron::calcDotProduct, ::_1));
                auto end =
                 boost::make_transform_iterator(m_neurons.end(),
                                                boost::bind(&Neuron::calcDotProduct, ::_1));
                using IteratorType = decltype(begin);
                std::for_each(m_neurons.begin(),
                              m_neurons.end(),
                              std::bind(&Neuron::template calculateOutput< IteratorType >,
                                        std::placeholders::_1,
                                        begin,
                                        end));
            }
        };
    } // namespace detail

    template< template< template< class > class, class, std::size_t, int > class NeuronType,
              template< class > class ActivationFunctionType,
              std::size_t size,
              std::size_t inputsNumber = 2,
              int scaleFactor = 1,
              typename Var = float >
    using NeuralLayer =
     detail::NeuralLayer< NeuronType< ActivationFunctionType, Var, inputsNumber, scaleFactor >, size, inputsNumber >;
} // namespace nn

#endif
