#pragma once

#include <array>
#include <tuple>

#include <cereal/cereal.hpp>

namespace nn::bp {

template< typename Var, typename LayersTuple >
struct BPContext;

template< typename Var, typename... Layers >
struct BPContext< Var, std::tuple< Layers... > > {
    using Forward = std::tuple<std::array<Var, Layers::size()>...>;
    using Gradients = std::tuple<std::array<Var, Layers::size() * Layers::inputs()>...>;

    Forward outputs;
    Gradients weights;
    Forward biases;
    Forward deltas;
    Forward biasGradients;
    Gradients weightGradients;
};

template< typename Archive, typename Var, typename... Layers >
void serialize(Archive& ar, BPContext< Var, std::tuple< Layers... > >& ctx) {
    ar(cereal::make_nvp("outputs", ctx.outputs));
    ar(cereal::make_nvp("weights", ctx.weights));
    ar(cereal::make_nvp("biases", ctx.biases));
    ar(cereal::make_nvp("deltas", ctx.deltas));
    ar(cereal::make_nvp("biasGradients", ctx.biasGradients));
    ar(cereal::make_nvp("weightGradients", ctx.weightGradients));
}

} // namespace nn::bp
