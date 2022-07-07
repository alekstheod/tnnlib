#pragma once

#include <NeuralNetwork/NeuralLayer/Vector.h>

#include <MPL/Algorithm.h>

#include <range/v3/all.hpp>

#include <algorithm>
#include <array>
#include <functional>

namespace nn {

    namespace detail {
        template< typename T >
        struct NeuralLayer;

        /**
         * Represent the NeuralLayer in perceptron.
         */
        template< typename T, std::size_t neuronsNumber >
        struct NeuralLayer< Vector< T, neuronsNumber > >
         : private Layer< Vector< T, neuronsNumber > > {
            using Base = Layer< Vector< T, neuronsNumber > >;
            using Container = typename Base::Container;
            using Var = typename Base::Var;
            using Memento = typename Base::Memento;

            template< template< class > typename NewType >
            using wrap = typename Base::template wrap_neuron< NeuralLayer, NewType >;

            template< unsigned int inputs >
            using adjust = typename Base::template adjust_inputs< NeuralLayer, inputs >;

            template< typename VarType >
            using use = typename Base::template use_var< NeuralLayer, VarType >;

            using Base::begin;
            using Base::cbegin;
            using Base::cend;
            using Base::end;
            using Base::size;
            using Base::operator[];
            using Base::getOutput;

            static constexpr unsigned int CONST_NEURONS_NUMBER = neuronsNumber;
            static constexpr unsigned int CONST_INPUTS_NUMBER = T::size();

            static_assert(neuronsNumber > 0,
                          "Invalid template argument neuronsNumber == 0");
            static_assert(T::size() > 1,
                          "Invalid template argument inputsNumber <= 1");

            template< typename Func >
            void for_each(Func func) {
                utils::for_< size() >([this, &func](auto i) {
                    func(i, utils::get< i.value >(m_neurons));
                });
            }

            template< typename Func >
            void for_each(Func func) const {
                utils::for_< size() >([this, &func](auto i) {
                    func(i, utils::get< i.value >(m_neurons));
                });
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

            template< typename Layer >
            void calculateOutputs(Layer& nextLayer) {
                std::array< Var, size() > dotProducts;
                utils::for_< size() >([this, &dotProducts](auto i) {
                    dotProducts[i.value] =
                     utils::get< i.value >(m_neurons).calcDotProduct();
                });

                utils::for_< size() >([this, &dotProducts, &nextLayer](auto i) {
                    const auto output = utils::get< i.value >(m_neurons).calculateOutput(
                     std::cbegin(dotProducts), std::cend(dotProducts));
                    nextLayer.setInput(i.value, output);
                });
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
     detail::NeuralLayer< nn::detail::Vector< NeuronType< ActivationFunctionType, Var, inputsNumber >, size > >;
} // namespace nn
