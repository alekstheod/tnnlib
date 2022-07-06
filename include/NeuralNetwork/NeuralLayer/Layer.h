#pragma once

#include <NeuralNetwork/Serialization/NeuralLayerMemento.h>

#include <MPL/Algorithm.h>


namespace nn {

    template< typename T, std::size_t sz = utils::size_of(T{}) >
    struct Layer {
        using Container = T;
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
} // namespace nn
