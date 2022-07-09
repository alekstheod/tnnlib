#pragma once

#include <NeuralNetwork/NeuralLayer/Container.h>
#include <NeuralNetwork/NeuralLayer/Vector.h>
#include <NeuralNetwork/NeuralLayer/Tuple.h>

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
        template< typename T >
        struct NeuralLayer : private Layer< T > {
            using Base = Layer< T >;
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
            using Base::inputs;
            using Base::size;
            using Base::operator[];

            static constexpr unsigned int CONST_NEURONS_NUMBER = size();
            static constexpr unsigned int CONST_INPUTS_NUMBER = inputs();

            static_assert(size() > 0,
                          "Invalid template argument neuronsNumber == 0");
            static_assert(inputs() >= 1,
                          "Invalid template argument inputsNumber <= 1");

            const Var& getOutput(unsigned int outputId) const {
                return self[outputId].getOutput();
            }

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
                utils::for_< size() >([this, &dotProducts](auto i) {
                    dotProducts[i.value] =
                     utils::get< i.value >(m_neurons).calcDotProduct();
                });

                for_each([&dotProducts](auto, auto& neuron) {
                    neuron.calculateOutput(std::cbegin(dotProducts), std::cend(dotProducts));
                });
            }

          private:
            using Base::m_neurons;
            NeuralLayer& self{*this};
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

    template< std::size_t inputs, typename Var, typename... Neuron >
    using ComplexNeuralInputLayer =
     detail::NeuralLayer< detail::Tuple< Var, inputs, typename Neuron::template adjust< inputs >... > >;

    template< typename... Neuron >
    using ComplexNeuralLayer =
     detail::NeuralLayer< detail::Tuple< float, 1, Neuron... > >;
} // namespace nn
