#pragma once

#include "NeuralNetwork/Serialization/PerceptronMemento.h"

#include <cereal/cereal.hpp>

namespace nn {

    template< typename Var >
    struct ComplexLayerMemento {
        template< class Archive >
        void save(Archive& archive) const {
            archive(CEREAL_NVP(perceptron));
        }

        template< class Archive >
        void load(Archive& archive) {
            archive(CEREAL_NVP(perceptron));
        }

        PerceptronMemento< Var > perceptron;
    };
} // namespace nn
