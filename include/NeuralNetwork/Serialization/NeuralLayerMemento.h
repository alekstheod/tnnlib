#pragma once

#include <NeuralNetwork/Serialization/NeuronMemento.h>

#include <cereal/cereal.hpp>

#include <vector>

namespace nn {

    template< typename NeuronMemento, std::size_t neuronsNumber >
    struct NeuralLayerMemento {
        using Container = std::vector< NeuronMemento >;
        NeuralLayerMemento() : neurons(neuronsNumber) {
        }

        template< class Archive >
        void save(Archive& archive) const {
            archive(CEREAL_NVP(neurons));
        }

        template< class Archive >
        void load(Archive& archive) {
            archive(CEREAL_NVP(neurons));
        }

        Container neurons;
    };
} // namespace nn
