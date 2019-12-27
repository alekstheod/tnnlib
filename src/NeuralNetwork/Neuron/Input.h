#pragma once

#include <System/Time.h>

#include <cereal/cereal.hpp>

namespace nn {
    template< typename Var >
    struct Input {
        Input(const Var& w = utils::createRandom< Var >(1) / Var{100.f},
              const Var& v = Var{0})
         : weight(w), value(v) {
        }

        template< class Archive >
        void save(Archive& ar) const {
            ar(CEREAL_NVP(weight));
        }

        template< class Archive >
        void load(Archive& ar) {
            ar(CEREAL_NVP(weight));
        }

        Var weight;
        Var value;
    };
} // namespace nn
