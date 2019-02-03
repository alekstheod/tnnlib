#pragma once
#include <boost/numeric/conversion/cast.hpp>

namespace utils {

    namespace priv {
        template< typename Var >
        Var randomize(unsigned int maxValue);

        template< typename Var >
        Var rnd(unsigned int maxValue);
    } // namespace priv


    template< typename Var >
    Var createRandom(unsigned int maxValue) {
        if(maxValue <= 1) {
            return priv::randomize< Var >(100u) / boost::numeric_cast< Var >(100.f);
        } else {
            return priv::randomize< Var >(maxValue);
        }
    }
} // namespace utils
