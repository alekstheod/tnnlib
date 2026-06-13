#pragma once

#include "NeuralNetwork/Utils/Utils.h"

#include <MPL/Tuple.h>
#include <MPL/Algorithm.h>
#include <MPL/TypeTraits.h>

#include <tuple>

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

        template< typename Var, typename... L >
        struct Perceptron {
            using VarType = Var;

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

            template< typename T >
            using use = decltype(perceptron< T >(std::declval< Layers >()));

            using Input = typename InputLayerType::Input;
            using Context = std::tuple<std::array<Var, L::size()>...>;

          private:
            Layers m_layers;
            Context m_context;

            static_assert(std::tuple_size< Layers >::value > 1,
                          "Invalid number of layers, at least two layers need "
                          "to be set");

          public:
            Layers& layers() {
                return m_layers;
            }

            Context& context() {
                return m_context;
            }

            const Context& context() const {
                return m_context;
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
                auto& inputLayer = std::get< 0 >(m_layers);
                unsigned int inputId = 0;
                while(begin != end) {
                    for(std::size_t featureIdx = 0; featureIdx < begin->value.size();
                        ++featureIdx) {
                        inputLayer[inputId][featureIdx].weight = 1.f;
                        inputLayer[inputId].setBias({});
                        inputLayer[inputId][featureIdx].value = begin->value[featureIdx];
                    }
                    begin++;
                    inputId++;
                }

                auto& inputLayer0 = std::get< 0 >(m_layers);
                inputLayer0.template calculateOutputs< decltype(m_context), 0 >(m_context);

                utils::for_< size() - 1U >([this](auto i) {
                    auto& layer = utils::get< i.value + 1 >(m_layers);
                    layer.template calculateOutputs< decltype(m_context), i.value + 1, i.value >(m_context);
                });

                auto& outputCtx = std::get< size() - 1U >(m_context);
                for(const auto& val : outputCtx) {
                    *out = val;
                    ++out;
                }
            }
        };
    } // namespace detail

    template< typename Var, typename... NeuralLayers >
    using Perceptron = detail::Perceptron< Var, NeuralLayers... >;
} // namespace nn
