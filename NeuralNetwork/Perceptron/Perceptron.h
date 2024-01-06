#pragma once

#include "NeuralNetwork/Serialization/PerceptronMemento.h"
#include "NeuralNetwork/Utils/Utils.h"

#include <MPL/Tuple.h>
#include <MPL/Algorithm.h>
#include <MPL/TypeTraits.h>

#include <algorithm>
#include <cassert>
#include <tuple>
#include <type_traits>

namespace nn {

    namespace detail {
        template< typename Var, typename... Layers >
        struct Perceptron;

        template< typename Var, typename... Layers >
        static constexpr auto perceptron(std::tuple< Layers... >)
         -> Perceptron< Var, Layers... >;

        template< template< class > class Wrapper, typename... Layers >
        constexpr auto wrap_layers(std::tuple< Layers... >)
         -> std::tuple< Wrapper< Layers >... >;

        template< typename... Layers >
        constexpr auto layers_memento(std::tuple< Layers... >)
         -> std::tuple< typename Layers::Memento... >;

        /*! \class Perceptron
         *  \briefs Contains an input neurons layer one output and one or more
         * hidden layers.
         */
        template< typename VarType, typename... L >
        struct Perceptron {
            using Var = VarType;

          private:
            using TmplLayers = std::tuple< typename L::template use< Var >... >;

          public:
            using InputLayerType = utils::front_t< TmplLayers >;

            static constexpr auto size() {
                return sizeof...(L);
            }

            static constexpr auto inputs() {
                return InputLayerType::size();
            }

            using Layers = typename mpl::rebindInputs< TmplLayers >::type;

            using OutputLayerType =
             typename std::tuple_element< size() - 1, Layers >::type;

            static constexpr auto outputs() {
                return OutputLayerType::size();
            }

            template< template< class > typename Layer >
            using wrap =
             decltype(perceptron< Var >(wrap_layers< Layer >(std::declval< Layers >())));

            using LayersMemento = decltype(layers_memento(std::declval< Layers >()));
            using Memento = PerceptronMemento< LayersMemento >;

            template< typename T >
            using use = decltype(perceptron< T >(std::declval< Layers >()));

            using Input = typename InputLayerType::Input;

          private:
            Layers m_layers;

            static_assert(std::tuple_size< Layers >::value > 1,
                          "Invalid number of layers, at least two layers need "
                          "to be set");

          public:
            Layers& layers() {
                return m_layers;
            }

            void setMemento(const Memento& memento) {
                utils::for_< size() >([this, &memento](auto i) {
                    auto& layer = utils::get< i.value >(m_layers);
                    layer.setMemento(utils::get< i.value >(memento.layers));
                });
            }

            Memento getMemento() const {
                LayersMemento memento;
                utils::for_< size() >([this, &memento](auto i) {
                    auto& layer = utils::get< i.value >(m_layers);
                    utils::get< i.value >(memento) = layer.getMemento();
                });

                return {memento};
            }

            /*!
             * @brief this method will calculate the outputs of perceptron.
             * @param begin is the iterator which is pointing to the first input
             * @param end the iterator which is pointing to the last input
             * @param out the output iterator where the results of the
             * calculation will be stored.
             */
            template< typename Iterator, typename OutputIterator >
            void calculate(Iterator begin, Iterator end, OutputIterator out) {
                unsigned int inputId = 0;
                while(begin != end) {
                    std::get< 0 >(m_layers).setInput(inputId, *begin);
                    begin++;
                    inputId++;
                }

                utils::for_< size() - 1U >([this](auto i) {
                    auto& layer = utils::get< i.value >(m_layers);
                    auto& nextLayer = utils::get< i.value + 1 >(m_layers);
                    layer.calculateOutputs(nextLayer);
                });

                auto& lastLayer = utils::get< size() - 1U >(m_layers);
                lastLayer.calculateOutputs();

                for(const auto& neuron : lastLayer) {
                    *out = neuron.getOutput();
                    ++out;
                }
            }
        };
    } // namespace detail

    template< typename Var, typename... NeuralLayers >
    using Perceptron = detail::Perceptron< Var, NeuralLayers... >;
} // namespace nn
