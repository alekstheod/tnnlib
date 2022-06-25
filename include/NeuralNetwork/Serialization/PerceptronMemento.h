#pragma once

#include <cereal/cereal.hpp>

namespace nn {
    template< typename Layers >
    struct PerceptronMemento {
        template< class Archive >
        void save(Archive& ar) const {
            ar(CEREAL_NVP(layers));
        }

        template< class Archive >
        void load(Archive& ar) {
            ar(CEREAL_NVP(layers));
        }

        Layers layers;
    };
} // namespace nn
