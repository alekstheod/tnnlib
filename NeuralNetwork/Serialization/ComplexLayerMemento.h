#pragma once

#include "NeuralNetwork/Serialization/PerceptronMemento.h"

#include <cereal/cereal.hpp>

namespace nn {
    template< typename Var >
    struct ComplexLayerMemento {
        PerceptronMemento< Var > perceptron;
    };
} // namespace nn
