#pragma once

#include <cstddef>
#include <vector>

namespace nn::bp {

    struct ConnectionData {
        std::vector<std::vector<std::vector<std::size_t>>> layers;

        template< std::size_t NeuronId, typename... InputIds >
        void add(std::size_t layerIdx, InputIds... inputIds) {
            if(layers.size() <= layerIdx) {
                layers.resize(layerIdx + 1);
            }
            auto& layer = layers[layerIdx];
            if(layer.size() <= NeuronId) {
                layer.resize(NeuronId + 1);
            }
            layer[NeuronId] = {static_cast< std::size_t >(inputIds)...};
        }

        bool empty() const {
            for(const auto& layer : layers) {
                for(const auto& neuron : layer) {
                    if(!neuron.empty()) {
                        return false;
                    }
                }
            }
            return true;
        }
    };

} // namespace nn::bp
