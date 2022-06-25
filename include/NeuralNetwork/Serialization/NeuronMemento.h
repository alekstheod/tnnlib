#pragma once

#include <NeuralNetwork/Neuron/Input.h>

#include <range/v3/all.hpp>
#include <cereal/cereal.hpp>

#include <utility>
#include <vector>
#include <functional>

namespace nn {

    /**s
     * @author alekstheod
     * Represents the Neuron's memento (state)
     * class. The instance of this class is enough in order
     * to restore the Neuron's state.
     */
    template< class Var, std::size_t inputsNumber >
    struct NeuronMemento {
        template< class Archive >
        void save(Archive& ar) const {
            ar(CEREAL_NVP(bias));
            auto inputValues =
             inputs | ranges::views::transform(std::mem_fn(&Input< Var >::weight)) |
             ranges::to< std::vector >;
            ar(inputValues);
        }

        template< class Archive >
        void load(Archive& ar) {
            ar(CEREAL_NVP(bias));
            auto inputValues =
             inputs | ranges::views::transform(std::mem_fn(&Input< Var >::weight)) |
             ranges::to< std::vector >;
            ar(inputValues);
            std::transform(cbegin(inputValues), cend(inputValues), begin(inputs), [](const auto& weight) {
                return Input< Var >{weight, {}};
            });
        }

        using Inputs = std::array< nn::Input< Var >, inputsNumber >;
        Var bias;
        Inputs inputs;
    };
} // namespace nn
