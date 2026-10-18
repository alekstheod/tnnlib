#pragma once

#include <cstddef>
#include <iterator>
#include <limits>
#include <tuple>
#include <vector>

namespace nn {

    static constexpr std::size_t connectionSentinel = std::numeric_limits< std::size_t >::max();

    /// Identifies the source of a connection: output `neuron` of layer `layer`,
    /// stored as indices into BPContext::outputs. A disconnected connection
    /// reads nothing; a connection pointing to the same or a later layer reads
    /// the value left from the previous evaluation (recurrent state).
    struct Connection {
        std::size_t layer = connectionSentinel;
        std::size_t neuron = connectionSentinel;

        constexpr bool is_connected() const noexcept {
            return layer != connectionSentinel;
        }

        constexpr bool operator==(const Connection& other) const noexcept {
            return layer == other.layer && neuron == other.neuron;
        }

        constexpr bool operator!=(const Connection& other) const noexcept {
            return !(*this == other);
        }
    };

    constexpr Connection connect(std::size_t layer, std::size_t neuron) {
        return Connection{layer, neuron};
    }

    /// Resolves a connection against the tuple of per-layer outputs.
    /// Returns nullptr when disconnected or out of range.
    template< typename Var, typename... LayerOutputs >
    const Var* outputOf(const std::tuple< LayerOutputs... >& outputs, const Connection& src) {
        if(!src.is_connected()) {
            return nullptr;
        }
        const Var* result = nullptr;
        std::size_t idx = 0;
        std::apply(
         [&](const auto&... layers) {
             ((idx++ == src.layer ? void(result = src.neuron < std::size(layers)
                                                       ? layers.data() + src.neuron
                                                       : nullptr)
                                  : void()),
              ...);
         },
         outputs);
        return result;
    }

} // namespace nn

namespace nn::bp {

    /// Runtime description of the connection matrix.
    /// layers[targetLayer][targetNeuron][slot] = source connection.
    struct ConnectionData {
        std::vector< std::vector< std::vector< Connection > > > layers;

        void connect(std::size_t targetLayer,
                     std::size_t targetNeuron,
                     std::size_t slot,
                     Connection src) {
            if(layers.size() <= targetLayer) {
                layers.resize(targetLayer + 1);
            }
            auto& layer = layers[targetLayer];
            if(layer.size() <= targetNeuron) {
                layer.resize(targetNeuron + 1);
            }
            auto& slots = layer[targetNeuron];
            if(slots.size() <= slot) {
                slots.resize(slot + 1);
            }
            slots[slot] = src;
        }

        /// Assigns consecutive input slots of neuron `NeuronId` in layer
        /// `layerIdx`, starting from slot 0.
        template< std::size_t NeuronId, typename... Srcs >
        void add(std::size_t layerIdx, Srcs... srcs) {
            std::size_t slot = 0;
            (connect(layerIdx, NeuronId, slot++, srcs), ...);
        }

        bool empty() const {
            for(const auto& layer : layers) {
                for(const auto& neuron : layer) {
                    for(const auto& slot : neuron) {
                        if(slot.is_connected()) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
    };

} // namespace nn::bp
