#pragma once

#include <boost/numeric/conversion/cast.hpp>

#include <cereal/cereal.hpp>

namespace nn {

    template< typename Var >
    struct Input {
        Input(const Var& w = Var{0}, const Var& v = Var{0})
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
