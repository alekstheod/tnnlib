#pragma once

#include "NeuralNetwork/BackPropagation/BpMemento.h"

#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/tuple.hpp>

namespace nn::bp {

template< typename Archive, typename Var, typename... Layers >
void serialize(Archive& ar, BpMemento< Var, std::tuple< Layers... > >& m) {
    ar(cereal::make_nvp("weights", m.weights));
    ar(cereal::make_nvp("biases", m.biases));
}

} // namespace nn::bp
