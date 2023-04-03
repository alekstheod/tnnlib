#pragma once

#include <array>

namespace nn {
    /**
     * @author alekstheod
     * Represents the Neuron's memento (state)
     * class. The instance of this class is enough in order
     * to restore the Neuron's state.
     */
    template< class Var, std::size_t inputsNumber >
    struct NeuronMemento {
        Var bias;
        std::array< Var, inputsNumber > weights;
    };
} // namespace nn
