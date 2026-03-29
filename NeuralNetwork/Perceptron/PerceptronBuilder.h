#pragma once

#include "NeuralNetwork/Perceptron/Perceptron.h"
#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"

#include "Utilities/MPL/Tuple.h"

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <tuple>

namespace nn {

    template< typename VarType, typename CurrentLayer, typename... PrevLayers >
    struct PerceptronBuilder {

        template< std::size_t size >
        constexpr auto dense() const {
            using L = nn::NeuralLayer< Neuron, SigmoidFunction, size >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }

        template< typename... Neurons >
        constexpr auto dense_complex() const {
            using L = nn::ComplexNeuralLayer< Neurons... >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }

        template< typename N >
        constexpr auto with_neuron() const {
            using L = typename CurrentLayer::template with_neuron< N >;
            return PerceptronBuilder< VarType, L, PrevLayers... >{};
        }

        static constexpr std::size_t size() {
            return nn::Perceptron< VarType, PrevLayers..., CurrentLayer >::size();
        }

        using type = nn::Perceptron< VarType, PrevLayers..., CurrentLayer >;
    };

    template< typename VarType >
    struct PerceptronBuilderBegin {
        template< std::size_t size >
        constexpr auto input() const {
            using L = InputLayer< Neuron, SigmoidFunction, size >;
            return PerceptronBuilder< VarType, L >{};
        }
    };

    template< typename VarType = float >
    constexpr auto build() {
        return PerceptronBuilderBegin< VarType >{};
    }

} // namespace nn
