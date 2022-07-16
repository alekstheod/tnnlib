#pragma once

#include <NeuralNetwork/NeuralLayer/Container.h>
#include <NeuralNetwork/Serialization/NeuralLayerMemento.h>

#include <MPL/Algorithm.h>

#include <vector>


namespace nn {
    namespace detail {
        template< typename Container, std::size_t sz >
        struct Vector {
            static constexpr auto size() {
                return sz;
            }
        };

        template< typename T, std::size_t sz >
        struct Layer< detail::Vector< T, sz > > {
            using Container = std::vector< T >;
            using Neuron = T;
            using Var = typename Neuron::Var;
            using NeuronMemento = typename Neuron::Memento;
            using Memento = NeuralLayerMemento< NeuronMemento, sz >;

            static constexpr auto inputs() {
                return Neuron::size();
            }

            static constexpr auto size() {
                return sz;
            }

            template< template< class > typename L, template< class > class NewType >
            using wrap_neuron = L< Vector< NewType< Neuron >, sz > >;

            template< template< class > typename L, std::size_t inputs >
            using adjust_inputs =
             L< Vector< typename Neuron::template resize< inputs >, sz > >;

            template< template< class > typename L, typename V >
            using use_var = L< Vector< typename Neuron::template use< V >, sz > >;

            const auto& operator[](unsigned int id) const {
                return m_neurons[id];
            }

            auto& operator[](unsigned int id) {
                return m_neurons[id];
            }

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

            Container m_neurons{sz};
        };

    } // namespace detail
} // namespace nn
  //
