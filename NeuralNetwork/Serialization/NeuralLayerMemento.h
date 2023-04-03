#pragma once

#include "NeuralNetwork/Serialization/NeuronMemento.h"

#include <cereal/cereal.hpp>

#include <vector>

namespace nn {
    template< typename NeuronMemento, std::size_t neuronsNumber >
    struct NeuralLayerMemento {
        std::vector< NeuronMemento > neurons{neuronsNumber};
    };
} // namespace nn
