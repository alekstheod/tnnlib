#pragma once

#include "NeuralNetwork/Serialization/NeuronMemento.h"
#include "NeuralNetwork/Serialization/NeuralLayerMemento.h"
#include "NeuralNetwork/Serialization/PerceptronMemento.h"
#include "NeuralNetwork/Serialization/ComplexLayerMemento.h"

#include <range/v3/all.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>

namespace nn {
    template< typename Archive, class Var, std::size_t inputsNumber >
    void serialize(Archive& ar, nn::NeuronMemento< Var, inputsNumber >& neuron) {
        ar(cereal::make_nvp("bias", neuron.bias));
        ar(cereal::make_nvp("weights", neuron.weights));
    }

    template< typename Archive, typename NeuronMemento, std::size_t neuronsNumber >
    void serialize(Archive& ar, nn::NeuralLayerMemento< NeuronMemento, neuronsNumber >& layer) {
        ar(cereal::make_nvp("neurons", layer.neurons));
    }

    template< typename Archive, typename Layers >
    void serialize(Archive& ar, nn::PerceptronMemento< Layers >& perceptron) {
        ar(cereal::make_nvp("layers", perceptron.layers));
    }

    template< typename Archive, typename Perceptron >
    void serialize(Archive& ar, nn::ComplexLayerMemento< Perceptron >& complexLayer) {
        ar(cereal::make_nvp("perceptron", complexLayer.perceptron));
    }
} // namespace nn
