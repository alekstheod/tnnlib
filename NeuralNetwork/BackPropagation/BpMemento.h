#pragma once

#include "NeuralNetwork/BackPropagation/BPContext.h"

#include <array>
#include <tuple>

namespace nn::bp {

template< typename Var, typename LayersTuple >
struct BpMemento;

template< typename Var, typename... Layers >
struct BpMemento< Var, std::tuple< Layers... > > {
    using Biases = std::tuple<std::array< Var, Layers::size() >...>;
    using Weights = std::tuple<std::array< Var, Layers::size() * Layers::inputs() >...>;

    Weights weights;
    Biases biases;
};

} // namespace nn::bp
