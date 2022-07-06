#pragma once

#include <NeuralNetwork/Serialization/NeuralLayerMemento.h>

#include <MPL/Algorithm.h>


namespace nn {
    namespace detail {
        template< typename Container, std::size_t sz >
        struct Vector {
            static constexpr auto size() {
                return sz;
            }
        };
    } // namespace detail
      //
    template< typename T >
    struct Layer;

    template< typename T, std::size_t sz >
    struct Layer< detail::Vector< std::vector< T >, sz > > {
        using Container = std::vector< T >;
        using Neuron = typename Container::value_type;
        using Var = typename Neuron::Var;
        using NeuronMemento = typename Neuron::Memento;
        using Memento = NeuralLayerMemento< NeuronMemento, sz >;
        static constexpr auto inputs = Neuron::size();

        static constexpr auto size() {
            return sz;
        }

        template< template< class, std::size_t, std::size_t > typename L, template< class > class NewType >
        using wrap_layer = L< NewType< Neuron >, size(), inputs >;

        template< template< class, std::size_t, std::size_t > typename L, std::size_t inputs >
        using adjust_layer = L< Neuron, size(), inputs >;

        template< template< class, std::size_t, std::size_t > typename L, typename V >
        using use_var = L< typename Neuron::template use< V >, size(), inputs >;

        Var getOutput(unsigned int outputId) const {
            return m_neurons[outputId].getOutput();
        }

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

    template< typename... T >
    struct Layer< std::tuple< T... > > {
        using Container = std::tuple< T... >;

        Container m_neurons;
    };
} // namespace nn
