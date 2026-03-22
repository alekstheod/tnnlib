#pragma once

#include "NeuralNetwork/Serialization/NeuronMemento.h"

#include <cereal/cereal.hpp>

#include <array>

namespace nn {
    template< typename NeuronMemento, std::size_t neuronsNumber >
    struct NeuralLayerMemento {
        std::array< NeuronMemento, neuronsNumber > neurons;
    };
} // namespace nn
