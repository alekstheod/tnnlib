/**
*  Copyright (c) 2011, Alex Theodoridis
*  All rights reserved.

*  Redistribution and use in source and binary forms, with
*  or without modification, are permitted provided that the
*  following conditions are met:
*  Redistributions of source code must retain the above
*  copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above
*  copyright notice, this list of conditions and the following
*  disclaimer in the documentation and/or other materials
*  provided with the distribution.

*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
*  AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
*  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
*  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
*  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
*  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
*  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE,
*  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

#ifndef NEURONMEMENTO_H
#define NEURONMEMENTO_H

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

#endif // NEURONMEMENTO_H
