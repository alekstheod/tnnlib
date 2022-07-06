#pragma once

#include <NeuralNetwork/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/Serialization/PerceptronMemento.h>
#include <NeuralNetwork/Utils/Utils.h>

#include <MPL/Tuple.h>
#include <MPL/Algorithm.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace nn {

    namespace detail {

        /*! \class Perceptron
         *  \briefs Contains an input neurons layer one output and one or more
         * hidden layers.
         */
        template< typename VarType, typename LayerTypes, std::size_t inputs = 1 >
        class Perceptron {
          private:
            using TmplLayers = typename mpl::rebindVar< VarType, LayerTypes >::type;

          public:
            static constexpr std::size_t CONST_LAYERS_NUMBER =
             std::tuple_size< TmplLayers >::value;
            using InputLayerType = typename std::tuple_element< 0, TmplLayers >::type;

            /// @brief the number of inputs is taken out of the argument if and
            /// only if it is bigger than 1, otherwise the number of inputs from
            /// the input layer is used.
            static constexpr std::size_t CONST_INPUTS_NUMBER =
             (inputs > 1) ? inputs : InputLayerType::CONST_INPUTS_NUMBER;
            using Layers =
             typename mpl::rebindInputs< CONST_INPUTS_NUMBER, TmplLayers >::type;
            using OutputLayerType =
             typename std::tuple_element< CONST_LAYERS_NUMBER - 1, Layers >::type;

            static constexpr std::size_t CONST_OUTPUTS_NUMBER =
             OutputLayerType::CONST_NEURONS_NUMBER;

            using Var = VarType;

            template< template< class > class Layer >
            using wrap =
             detail::Perceptron< VarType, typename utils::rebind_tuple< Layer, Layers >::type >;

            using LayersMemento = typename detail::mpl::ToMemento< Layers >::type;
            typedef PerceptronMemento< LayersMemento > Memento;

            template< typename T >
            using use = Perceptron< T, LayerTypes >;

            template< std::size_t in >
            using resize = Perceptron< VarType, LayerTypes, in >;
            using reverse = Perceptron< VarType, utils::reverse< LayerTypes > >;

          private:
            /*!
             * Hidden layers.
             */
            Layers m_layers;

            template< std::size_t index >
            void getMem(LayersMemento& layers) const {
                std::get< index >(layers) = std::get< index >(m_layers).getMemento();
                if constexpr(index < CONST_LAYERS_NUMBER - 1) {
                    getMem< index + 1 >(layers);
                }
            }

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

            static constexpr auto size() {
                return CONST_LAYERS_NUMBER;
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

    template< typename VarType, typename... NeuralLayers >
    using Perceptron = detail::Perceptron< VarType, std::tuple< NeuralLayers... > >;
} // namespace nn
