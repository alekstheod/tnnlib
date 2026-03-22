#pragma once

#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"

#include <range/v3/view.hpp>

namespace nn {
    namespace detail {

        template< typename Var, std::size_t featuresNumber >
        struct InputData {
            std::array< Var, featuresNumber > value;
        };

        template< typename Internal >
        struct InputLayer : private Internal {
            using Var = typename Internal::Var;
            using Memento = typename Internal::Memento;

            using Internal::begin;
            using Internal::cbegin;
            using Internal::cend;
            using Internal::end;
            using Internal::size;
            using Internal::operator[];
            using Internal::calculateOutputs;
            using Internal::for_each;
            using Internal::getMemento;
            using Internal::getOutput;
            using Internal::inputs;
            using Internal::setMemento;

            template< template< class > class NewType >
            using wrap = InputLayer< typename Internal::template wrap< NewType > >;

            template< unsigned int inputs >
            using adjust = InputLayer< typename Internal::template adjust< inputs > >;

            template< typename VarType >
            using use = InputLayer< typename Internal::template use< VarType > >;

            using Input = InputData< Var, inputs() >;

            void setInput(int neuronId, const Input& input) {
                auto& self = *this;
                for(const auto& featureId : ranges::views::indices(input.value.size())) {
                    self[neuronId][featureId].weight = 1.f;
                    self[neuronId].setBias({});
                    self[neuronId][featureId].value = input.value[featureId];
                }
            }
        };
    } // namespace detail

    /// @brief input neural layer
    /// @param NeuronType a type of the neuron in a layer.
    /// @param ActivationFunction a type of the activation function used in a
    /// neuron.
    /// @param size ammount of neurons in a layer.
    /// @param featuresNumber the number of features of each input in a layer.
    /// initialization a final weight will be calculated in a following way
    /// random(0, 1)/scaleFactor
    template< template< template< class > class, class, std::size_t > class NeuronType,
              template< class > class ActivationFunctionType,
              std::size_t size,
              std::size_t featuresNumber = 1,
              typename Var = float >
    using InputLayer =
     detail::InputLayer< NeuralLayer< NeuronType, ActivationFunctionType, size, featuresNumber > >;

} // namespace nn
