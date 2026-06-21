#pragma once

#include "NeuralNetwork/Perceptron/Perceptron.h"
#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"
#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/PoolingLayer.h"
#include "NeuralNetwork/NeuralLayer/Thread/AsyncNeuralLayer.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/Neuron/PoolingNeuron.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"

#include "Utilities/MPL/Tuple.h"

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <tuple>

namespace nn {

    template< std::size_t W, std::size_t H, typename K >
    struct SlidingWindowConfig {
        static constexpr std::size_t width = W;
        static constexpr std::size_t height = H;
        using Kernel = K;
    };

    template< typename VarType, typename CurrentLayer, typename... PrevLayers >
    struct PerceptronBuilder;

    template< typename VarType, typename ConvConfig, typename CurrentLayer, typename... PrevLayers >
    struct ConvBuilder {
        template< std::size_t W, std::size_t H, std::size_t S >
        constexpr auto with_kernel() const {
            using NewConfig =
             SlidingWindowConfig< ConvConfig::width, ConvConfig::height, nn::Kernel< W, H, S > >;
            return ConvBuilder< VarType, NewConfig, CurrentLayer, PrevLayers... >{};
        }

        template< std::size_t W, std::size_t H >
        constexpr auto with_grid() const {
            using NewConfig = SlidingWindowConfig< W, H, typename ConvConfig::Kernel >;
            return ConvBuilder< VarType, NewConfig, CurrentLayer, PrevLayers... >{};
        }

        constexpr auto build() const {
            using Grid =
             nn::SlidingWindow< ConvConfig::width, ConvConfig::height, typename ConvConfig::Kernel >;
            using L =
             ConvolutionLayer< nn::NeuralLayer, Neuron, SigmoidFunction, Grid, VarType >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }
    };

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

        template< typename SlidingWindow >
        constexpr auto conv() const {
            using L =
             ConvolutionLayer< nn::NeuralLayer, Neuron, SigmoidFunction, SlidingWindow, VarType >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }

        constexpr auto conv() const {
            using DefaultConfig = SlidingWindowConfig< 8, 8, nn::Kernel< 3, 3, 1 > >;
            return ConvBuilder< VarType, DefaultConfig, CurrentLayer, PrevLayers... >{};
        }

        template< template< class > class PoolingAlgo, typename SlidingWindow >
        constexpr auto pool() const {
            using L = PoolingLayer< nn::NeuralLayer, PoolingAlgo, SlidingWindow, VarType >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }

        template< std::size_t size >
        constexpr auto async() const {
            using L = nn::AsyncNeuralLayer< Neuron, SigmoidFunction, size >;
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
