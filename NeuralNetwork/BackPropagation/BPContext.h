#pragma once

#include <ConnectionData.h>

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>

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
    using Connections = std::tuple<std::array<Connection, Layers::size() * Layers::inputs()>...>;

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

    /// Applies the slots covered by @p data; slots without an entry keep the
    /// connections they already have.
    void applyConnections(const ConnectionData& data) {
        if(data.empty()) {
            return;
        }
        applyLayerConnections(data, std::make_index_sequence< sizeof...(Layers) >{});
    }

    /// Connection data with only the slots that deviate from the regular
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
        for(std::size_t n = 0; n < Layer::size(); ++n) {
            for(std::size_t s = 0; s < numInputs; ++s) {
                layer[n * numInputs + s] = Layer::defaultConnection(LayerIdx, n, s);
            }
        }
    }

    template< std::size_t... Is >
    void applyLayerConnections(const ConnectionData& data, std::index_sequence< Is... >) {
        (applyNeuronConnections< Is >(data), ...);
    }

    template< std::size_t LayerIdx >
    void applyNeuronConnections(const ConnectionData& data) {
        using Layer = std::tuple_element_t< LayerIdx, std::tuple< Layers... > >;
        constexpr auto numInputs = Layer::inputs();
        auto& target = std::get< LayerIdx >(connections);
        for(const auto& entry : data.entries) {
            if(entry.layer != LayerIdx || entry.neuron >= Layer::size() ||
               entry.slot >= numInputs) {
                continue;
            }
            target[entry.neuron * numInputs + entry.slot] = entry.source;
        }
    }

    template< std::size_t... Is >
    void collectLayerConnections(ConnectionData& result, std::index_sequence< Is... >) const {
        (collectNeuronConnections< Is >(result), ...);
    }

    template< std::size_t LayerIdx >
    void collectNeuronConnections(ConnectionData& result) const {
        using Layer = std::tuple_element_t< LayerIdx, std::tuple< Layers... > >;
        constexpr auto numInputs = Layer::inputs();
        const auto& source = std::get< LayerIdx >(connections);
        for(std::size_t neuronId = 0; neuronId < Layer::size(); ++neuronId) {
            for(std::size_t s = 0; s < numInputs; ++s) {
                const auto& connection = source[neuronId * numInputs + s];
                if(connection != Layer::defaultConnection(LayerIdx, neuronId, s)) {
                    result.connect(LayerIdx, neuronId, s, connection);
                }
            }
        }
    }
};

/// Serializes one connection override.
template< typename Archive >
void serialize(Archive& ar, ConnectionData::Entry& entry) {
    ar(cereal::make_nvp("layer", entry.layer));
    ar(cereal::make_nvp("neuron", entry.neuron));
    ar(cereal::make_nvp("slot", entry.slot));
    ar(cereal::make_nvp("source", entry.source));
}

/// Serializes the sparse entries of the connection matrix; a slot without an
/// entry uses the regular connection.
template< typename Archive >
void serialize(Archive& ar, ConnectionData& data) {
    ar(data.entries);
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
