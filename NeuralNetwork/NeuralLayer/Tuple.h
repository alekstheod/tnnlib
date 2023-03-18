#pragma once

#include "NeuralNetwork/NeuralLayer/Container.h"
#include "NeuralNetwork/Neuron/INeuron.h"
#include "NeuralNetwork/Serialization/NeuralLayerMemento.h"

#include <MPL/Algorithm.h>
#include <MPL/Utils.h>
#include <MPL/TypeTraits.h>

#include <tuple>

namespace nn {
    namespace detail {
        template< typename, std::size_t, typename... >
        struct Tuple {};

        template< typename VarType, std::size_t inputsNumber, typename... T >
        struct Layer< Tuple< VarType, inputsNumber, T... > > {
            using Var = VarType;

            template< template< class > typename L, template< class > class NewType >
            using wrap_neuron = L< Tuple< VarType, inputsNumber, NewType< T >... > >;

            template< template< class > typename L, typename V >
            using use_var = L< Tuple< V, inputsNumber, T... > >;

            template< template< class > typename L, std::size_t newInputs >
            using adjust_inputs =
             L< Tuple< Var, newInputs, typename T::template resize< newInputs >... > >;

            using Container = std::tuple< T... >;

            using Memento =
             NeuralLayerMemento< NeuronMemento< Var, inputsNumber >, sizeof...(T) >;

            static constexpr auto inputs() {
                return inputsNumber;
            }

            static constexpr auto size() {
                return sizeof...(T);
            }

            static constexpr unsigned int CONST_INPUTS_NUMBER = inputsNumber;

            Layer() : m_neurons{} {
                utils::for_< size() >([this](auto i) {
                    m_neuron_interfaces[i.value] = &(std::get< i.value >(m_neurons));
                });
            }

            auto& operator[](std::size_t idx) {
                return utils::deref(m_neuron_interfaces[idx]);
            }

            const auto& operator[](std::size_t idx) const {
                return utils::deref(m_neuron_interfaces[idx]);
            }

            auto cbegin() const {
                return std::cbegin(m_neuron_interfaces);
            }

            auto cend() const {
                return std::cend(m_neuron_interfaces);
            }

            auto begin() {
                return std::begin(m_neuron_interfaces);
            }

            auto end() {
                return std::end(m_neuron_interfaces);
            }

          protected:
            std::tuple< T... > m_neurons;

          private:
            std::array< INeuron< Var >*, size() > m_neuron_interfaces;
        };
    } // namespace detail
} // namespace nn
