#pragma once

#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"
#include "NeuralNetwork/Neuron/PoolingNeuron.h"

namespace nn {

    template< template< template< template< class > class, class, std::size_t > class, template< class > class, std::size_t size, std::size_t inputsNumber = 2, typename Var = float >
              typename NeuralLayerType,
              template< class >
              class PoolingAlgo,
              typename Grid,
              typename Var = float >
    using PoolingLayer =
     detail::ConvolutionLayer< NeuralLayerType< PoolingNeuron, PoolingAlgo, Grid::framesNumber, Grid::K::size >, Grid >;
}
