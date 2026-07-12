#pragma once

#include <MPL/Algorithm.h>

#include <array>
#include <tuple>
#include <limits>

#include <cereal/cereal.hpp>

namespace nn::bp {

static constexpr std::size_t connectionSentinel = std::numeric_limits< std::size_t >::max();

template< std::size_t N >
constexpr std::array< std::size_t, N > defaultConnectionIds() {
    std::array< std::size_t, N > result{};
    for(std::size_t i = 0; i < N; ++i) {
        result[i] = i;
    }
    return result;
}

template< typename Var, typename LayersTuple >
struct BPContext;

template< typename Var, typename... Layers >
struct BPContext< Var, std::tuple< Layers... > > {
    using Forward = std::tuple<std::array<Var, Layers::size()>...>;
    using Gradients = std::tuple<std::array<Var, Layers::size() * Layers::inputs()>...>;
    using Connections = std::tuple<std::array<std::array<std::size_t, Layers::inputs()>, Layers::size()>...>;

    Forward outputs{};
    Gradients weights{};
    Forward biases{};
    Forward deltas{};
    Forward biasGradients{};
    Gradients weightGradients{};
    Connections connections{ defaultConnectionTuple() };

  private:
    static Connections defaultConnectionTuple() {
        Connections result{};
        initLayerConnections(result, std::make_index_sequence<sizeof...(Layers)>{});
        return result;
    }

    template< std::size_t... Is >
    static void initLayerConnections(Connections& result, std::index_sequence< Is... >) {
        (initNeuronConnections< Is >(result), ...);
    }

    template< std::size_t LayerIdx >
    static void initNeuronConnections(Connections& result) {
        constexpr auto numInputs = std::tuple_element_t< LayerIdx, std::tuple< Layers... > >::inputs();
        constexpr auto defaultIds = defaultConnectionIds< numInputs >();
        auto& layer = std::get< LayerIdx >(result);
        for(std::size_t n = 0; n < layer.size(); ++n) {
            layer[n] = defaultIds;
        }
    }
};

template< typename Archive, typename Var, typename... Layers >
void serialize(Archive& ar, BPContext< Var, std::tuple< Layers... > >& ctx) {
    ar(cereal::make_nvp("weights", ctx.weights));
    ar(cereal::make_nvp("biases", ctx.biases));
}

} // namespace nn::bp
