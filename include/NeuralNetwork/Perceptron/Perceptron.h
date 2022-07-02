#pragma once

#include <NeuralNetwork/INeuralLayer.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/Serialization/PerceptronMemento.h>
#include <NeuralNetwork/Utils/Utils.h>

#include <MPL/Tuple.h>

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
            void setMem(const LayersMemento& layers) {
                std::get< index >(m_layers).setMemento(std::get< index >(layers));
                if constexpr(index < CONST_LAYERS_NUMBER - 1) {
                    setMem< index + 1 >(layers);
                }
            }

            template< std::size_t index >
            void getMem(LayersMemento& layers) const {
                std::get< index >(layers) = std::get< index >(m_layers).getMemento();
                if constexpr(index < CONST_LAYERS_NUMBER - 1) {
                    getMem< index + 1 >(layers);
                }
            }

            template< unsigned int index >
            void calculate(Layers& layers) {
                std::get< index >(layers).calculateOutputs(std::get< index + 1 >(layers));
                if constexpr(index < CONST_LAYERS_NUMBER - 2) {
                    calculate< index + 1 >(layers);
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
                setMem< 0 >(memento.layers);
            }

            Memento getMemento() const {
                LayersMemento layers;
                getMem< 0 >(layers);
                return Memento{layers};
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

                calculate< 0 >(m_layers);
                using OutputLayer =
                 typename std::tuple_element< CONST_LAYERS_NUMBER - 1, Layers >::type;
                std::get< CONST_LAYERS_NUMBER - 1 >(m_layers).calculateOutputs();
                std::transform(std::get< CONST_LAYERS_NUMBER - 1 >(m_layers).begin(),
                               std::get< CONST_LAYERS_NUMBER - 1 >(m_layers).end(),
                               out,
                               std::bind(&OutputLayer::Neuron::getOutput,
                                         std::placeholders::_1));
            }

            /**
             * @brief only for the testing purpose.
             * @brief please don't use this function.
             */
            template< typename Test >
            void supportTest(Test&);
        };
    } // namespace detail

    template< typename VarType, typename... NeuralLayers >
    using Perceptron = detail::Perceptron< VarType, std::tuple< NeuralLayers... > >;
} // namespace nn
