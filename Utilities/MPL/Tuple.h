#pragma once

#include "Utilities/MPL/Algorithm.h"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>


namespace utils {
    template< typename Functor, typename... ArgsT >
    void for_each(std::tuple< ArgsT... >& t, Functor func) {
        for_< sizeof...(ArgsT) >([&](auto i) { func(std::get< i.value >(t)); });
    }

    template< typename LastElement, typename... Elements >
    struct push_back;

    template< typename LastElement, typename... Elements >
    struct push_back< LastElement, std::tuple< Elements... > > {
        typedef typename std::tuple< Elements..., LastElement > type;
    };

    template< typename LastElement, typename... Elements >
    using push_back_t = push_back< LastElement, Elements... >;

    template< typename Tuple >
    using back_t = std::tuple_element_t< std::tuple_size< Tuple >::value - 1, Tuple >;

    template< typename Tuple >
    using front_t = std::tuple_element_t< 0, Tuple >;
} // namespace utils
