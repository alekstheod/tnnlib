#pragma once

#include <ConnectionData.h>

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include <cereal/cereal.hpp>

namespace nn {

/// Serializes a connection source for BPContext archives.
template< typename Archive >
void serialize(Archive& ar, Connection& src) {
    ar(cereal::make_nvp("layer", src.layer));
    ar(cereal::make_nvp("neuron", src.neuron));
}

} // namespace nn

namespace nn::bp {

template< typename Var, typename LayersTuple >
struct BPContext;

template< typename Var, typename... Layers >
struct BPContext< Var, std::tuple< Layers... > > {
    using Forward = std::tuple<std::array<Var, Layers::size()>...>;
    using Gradients = std::tuple<std::array<Var, Layers::size() * Layers::inputs()>...>;
    using Connections = std::tuple<std::array<std::array<Connection, Layers::inputs()>, Layers::size()>...>;

    Forward outputs{};
    Gradients weights{};
    Forward biases{};
    Forward deltas{};
    Forward biasGradients{};
    Gradients weightGradients{};
    Connections connections{ defaultConnectionTuple() };

    /// Restores the regular connection for every neuron and input slot.
    void resetConnections() {
        connections = defaultConnectionTuple();
    }

    /// Wires the neurons covered by @p data; neurons without an entry keep
    /// the connections they already have.
    void applyConnections(const ConnectionData& data) {
        if(data.empty()) {
            return;
        }
        applyLayerConnections(data, std::make_index_sequence< sizeof...(Layers) >{});
    }

    /// Connection data with only the neurons that deviate from the regular
    /// connections; this is what gets serialized.
    ConnectionData connectionOverrides() const {
        ConnectionData result;
        collectLayerConnections(result, std::make_index_sequence< sizeof...(Layers) >{});
        return result;
    }

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
        using Layer = std::tuple_element_t< LayerIdx, std::tuple< Layers... > >;
        constexpr auto numInputs = Layer::inputs();
        auto& layer = std::get< LayerIdx >(result);
        for(std::size_t n = 0; n < layer.size(); ++n) {
            for(std::size_t s = 0; s < numInputs; ++s) {
                layer[n][s] = Layer::defaultConnection(LayerIdx, n, s);
            }
        }
    }

    template< std::size_t... Is >
    void applyLayerConnections(const ConnectionData& data, std::index_sequence< Is... >) {
        (applyNeuronConnections< Is >(data), ...);
    }

    template< std::size_t LayerIdx >
    void applyNeuronConnections(const ConnectionData& data) {
        if(LayerIdx >= data.layers.size()) {
            return;
        }
        const auto& layer = data.layers[LayerIdx];
        auto& target = std::get< LayerIdx >(connections);
        for(std::size_t neuronId = 0;
            neuronId < layer.size() && neuronId < target.size();
            ++neuronId) {
            const auto& src = layer[neuronId];
            if(src.empty()) {
                continue;
            }
            for(std::size_t s = 0; s < target[neuronId].size(); ++s) {
                target[neuronId][s] = s < src.size() ? src[s] : Connection{};
            }
        }
    }

    template< std::size_t... Is >
    void collectLayerConnections(ConnectionData& result, std::index_sequence< Is... >) const {
        (collectNeuronConnections< Is >(result), ...);
    }

    template< std::size_t LayerIdx >
    void collectNeuronConnections(ConnectionData& result) const {
        using Layer = std::tuple_element_t< LayerIdx, std::tuple< Layers... > >;
        const auto& source = std::get< LayerIdx >(connections);
        for(std::size_t neuronId = 0; neuronId < source.size(); ++neuronId) {
            const auto& neuron = source[neuronId];
            bool customized = false;
            for(std::size_t s = 0; s < neuron.size(); ++s) {
                if(neuron[s] != Layer::defaultConnection(LayerIdx, neuronId, s)) {
                    customized = true;
                    break;
                }
            }
            if(!customized) {
                continue;
            }
            for(std::size_t s = 0; s < neuron.size(); ++s) {
                result.connect(LayerIdx, neuronId, s, neuron[s]);
            }
        }
    }
};

/// Serializes the entries of the connection matrix; a neuron without an
/// entry uses the regular connection.
template< typename Archive >
void serialize(Archive& ar, ConnectionData& data) {
    ar(data.layers);
}

template< typename Archive, typename Var, typename... Layers >
void serialize(Archive& ar, BPContext< Var, std::tuple< Layers... > >& ctx) {
    ar(cereal::make_nvp("weights", ctx.weights));
    ar(cereal::make_nvp("biases", ctx.biases));
    if constexpr(std::is_same_v< typename Archive::is_loading, std::true_type >) {
        ConnectionData entries;
        try {
            ar(cereal::make_nvp("connections", entries));
        } catch(const cereal::Exception&) {
            // Models saved before the connection matrix existed carry no
            // entries; every neuron falls back to its regular connection.
        }
        ctx.resetConnections();
        ctx.applyConnections(entries);
    } else {
        auto entries = ctx.connectionOverrides();
        ar(cereal::make_nvp("connections", entries));
    }
}

} // namespace nn::bp
