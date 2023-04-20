#pragma once

#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"
#include "NeuralNetwork/Neuron/PoolingNeuron.h"

namespace nn {

    namespace detail {
        template< typename Internal >
        struct PoolingLayer : private Internal {
            using Internal::begin;
            using Internal::cbegin;
            using Internal::cend;
            using Internal::end;
            using Internal::for_each;
            using Internal::inputs;
            using Internal::setInput;
            using Internal::size;
            using Internal::operator[];
            using Internal::calculateOutputs;
            using Memento = typename Internal::Memento;
            using Internal::getMemento;
            using Internal::setMemento;
            // We can't adjust this layer as the
            // number of inputs and neurons depends
            // on the convolution grid and frame sizes
            template< unsigned int inputs >
            using adjust = PoolingLayer;

            template< typename VarType >
            using use = PoolingLayer< typename Internal::template use< VarType > >;
        };
    } // namespace detail

    template< template< template< template< class > class, class, std::size_t > class, template< class > class, std::size_t size, std::size_t inputsNumber, typename Var = float >
              typename NeuralLayerType,
              template< class >
              class PoolingAlgo,
              typename Grid,
              typename Var = float >
    using PoolingLayer = detail::PoolingLayer<
     detail::ConvolutionLayer< NeuralLayerType< PoolingNeuron, PoolingAlgo, Grid::framesNumber, Grid::K::size >, Grid > >;
} // namespace nn
