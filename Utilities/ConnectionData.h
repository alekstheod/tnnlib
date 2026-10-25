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

    /// Sparse description of the connection matrix.
    /// Every entry overrides exactly one input slot of one neuron. Slots
    /// without an entry keep the regular (default) connection of their layer.
    struct ConnectionData {
        struct Entry {
            std::size_t layer{connectionSentinel};
            std::size_t neuron{connectionSentinel};
            std::size_t slot{connectionSentinel};
            Connection source{};

            constexpr bool operator==(const Entry& other) const noexcept {
                return layer == other.layer && neuron == other.neuron &&
                       slot == other.slot && source == other.source;
            }

            constexpr bool operator!=(const Entry& other) const noexcept {
                return !(*this == other);
            }
        };

        std::vector< Entry > entries;

        void connect(std::size_t targetLayer,
                     std::size_t targetNeuron,
                     std::size_t slot,
                     Connection src) {
            for(auto& entry : entries) {
                if(entry.layer == targetLayer && entry.neuron == targetNeuron &&
                   entry.slot == slot) {
                    entry.source = src;
                    return;
                }
            }
            entries.push_back(Entry{targetLayer, targetNeuron, slot, src});
        }

        /// Assigns consecutive input slots of neuron `NeuronId` in layer
        /// `layerIdx`, starting from slot 0.
        template< std::size_t NeuronId, typename... Srcs >
        void add(std::size_t layerIdx, Srcs... srcs) {
            std::size_t slot = 0;
            (connect(layerIdx, NeuronId, slot++, srcs), ...);
        }

        /// Assigns a single input slot of neuron `NeuronId` in layer
        /// `layerIdx`; all other slots keep their default connection.
        template< std::size_t NeuronId, std::size_t Slot >
        void set(std::size_t layerIdx, Connection src) {
            connect(layerIdx, NeuronId, Slot, src);
        }

        bool empty() const {
            return entries.empty();
        }
    };

} // namespace nn::bp
