#pragma once

#include <array>
#include <tuple>

namespace nn::bp {

template< typename Var, typename LayersTuple >
struct BPContext;

template< typename Var, typename... Layers >
struct BPContext< Var, std::tuple< Layers... > > {
    using Forward = std::tuple<std::array<Var, Layers::size()>...>;
    using Gradients = std::tuple<std::array<Var, Layers::size() * Layers::inputs()>...>;

    Forward& outputs;
    Gradients weights;
    Forward biases;
    Forward deltas;
    Forward biasGradients;
    Gradients weightGradients;
};

} // namespace nn::bp
