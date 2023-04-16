#pragma once

#include <array>

namespace nn {
    template< class Var, std::size_t inputsNumber >
    struct NeuronMemento {
        Var bias;
        std::array< Var, inputsNumber > weights;
    };
} // namespace nn
