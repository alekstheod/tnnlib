#pragma once

#include <NeuralNetwork/NeuralLayer/Layer.h>

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
        template< typename NeuronType, std::size_t neuronsNumber, std::size_t inputsNumber >
        struct NeuralLayer
         : private Layer< std::vector< typename NeuronType::template adjust< inputsNumber > >, neuronsNumber > {
            using Base =
             Layer< std::vector< typename NeuronType::template adjust< inputsNumber > >, neuronsNumber >;
            using Neuron = typename Base::Neuron;
            using Container = typename Base::Container;
            using Var = typename Base::Var;
            using Memento = typename Base::Memento;

            template< template< class > typename NewType >
            using wrap = typename Base::template wrap_layer< NeuralLayer, NewType >;

            template< unsigned int inputs >
            using adjust = typename Base::template adjust_layer< NeuralLayer, inputs >;

            template< typename VarType >
            using use = typename Base::template use_var< NeuralLayer, VarType >;

            using Base::begin;
            using Base::cbegin;
            using Base::cend;
            using Base::end;
            using Base::size;

            static constexpr unsigned int CONST_NEURONS_NUMBER = neuronsNumber;
            static constexpr unsigned int CONST_INPUTS_NUMBER = inputsNumber;

            static_assert(neuronsNumber > 0,
                          "Invalid template argument neuronsNumber == 0");
            static_assert(inputsNumber > 1,
                          "Invalid template argument inputsNumber <= 1");

            const auto& operator[](unsigned int id) const {
                return m_neurons[id];
            }

            auto& operator[](unsigned int id) {
                return m_neurons[id];
            }

            void setInput(unsigned int inputId, const Var& value) {
                utils::for_< size() >([this, inputId, value](auto i) {
                    utils::get< i.value >(m_neurons).setInput(inputId, value);
                });
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
            using Base::m_neurons;
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
