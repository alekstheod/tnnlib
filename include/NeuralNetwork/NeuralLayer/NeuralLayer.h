#pragma once

#include <NeuralNetwork/Serialization/NeuralLayerMemento.h>

#include <MPL/Algorithm.h>

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
        struct NeuralLayer {
            using Neuron = typename NeuronType::template adjust< inputsNumber >;
            using Var = typename Neuron::Var;
            using NeuronMemento = typename Neuron::Memento;
            using Memento = NeuralLayerMemento< NeuronMemento, neuronsNumber >;

            template< template< class > class NewType >
            using wrap = NeuralLayer< NewType< Neuron >, neuronsNumber, inputsNumber >;

            template< unsigned int inputs >
            using adjust = NeuralLayer< Neuron, neuronsNumber, inputs >;

            template< typename VarType >
            using use =
             NeuralLayer< typename NeuronType::template use< VarType >, neuronsNumber, inputsNumber >;
            static constexpr unsigned int CONST_NEURONS_NUMBER = neuronsNumber;
            static constexpr unsigned int CONST_INPUTS_NUMBER = inputsNumber;
            typedef typename std::vector< Neuron > Container;

            static_assert(neuronsNumber > 0,
                          "Invalid template argument neuronsNumber == 0");
            static_assert(inputsNumber > 0,
                          "Invalid template argument inputsNumber <= 1");

            auto cbegin() const {
                return std::cbegin(m_neurons);
            }

            auto cend() const {
                return std::cend(m_neurons);
            }

            auto begin() {
                return std::begin(m_neurons);
            }

            auto end() {
                return std::end(m_neurons);
            }

            static constexpr auto size() {
                return CONST_NEURONS_NUMBER;
            }

            const Neuron& operator[](unsigned int id) const {
                return m_neurons[id];
            }

            Neuron& operator[](unsigned int id) {
                return m_neurons[id];
            }

            void setInput(unsigned int inputId, const Var& value) {
                std::for_each(std::begin(m_neurons),
                              std::end(m_neurons),
                              std::bind(&Neuron::setInput, std::placeholders::_1, inputId, value));
            }

            const Memento getMemento() const {
                Memento memento;
                utils::for_< size() >([this, &memento](auto i) {
                    memento.neurons[i.value] =
                     utils::get< i.value >(m_neurons).getMemento();
                });
                return memento;
            }

            void setMemento(const Memento& memento) {
                utils::for_< size() >([this, &memento](auto i) {
                    utils::get< i.value >(m_neurons).setMemento(memento.neurons[i.value]);
                });
            }

            Var getOutput(unsigned int outputId) const {
                return m_neurons[outputId].getOutput();
            }

            template< typename Layer >
            void calculateOutputs(Layer& nextLayer) {
                std::array< Var, size() > dotProducts;
                for(std::size_t i = 0U; i < size(); ++i) {
                    dotProducts[i] = m_neurons[i].calcDotProduct();
                }

                for(unsigned int i = 0; i < m_neurons.size(); i++) {
                    nextLayer.setInput(i,
                                       m_neurons[i].calculateOutput(std::cbegin(dotProducts),
                                                                    std::cend(dotProducts)));
                }
            }

            void calculateOutputs() {
                std::array< Var, size() > dotProducts;
                for(std::size_t i = 0U; i < size(); ++i) {
                    dotProducts[i] = m_neurons[i].calcDotProduct();
                }

                for(auto& neuron : m_neurons) {
                    neuron.calculateOutput(std::cbegin(dotProducts), std::cend(dotProducts));
                }
            }

          private:
            Container m_neurons{neuronsNumber};
        };
    } // namespace detail

    template< template< template< class > class, class, std::size_t > class NeuronType,
              template< class >
              class ActivationFunctionType,
              std::size_t size,
              std::size_t inputsNumber = 2,
              typename Var = float >
    using NeuralLayer =
     detail::NeuralLayer< NeuronType< ActivationFunctionType, Var, inputsNumber >, size, inputsNumber >;
} // namespace nn
