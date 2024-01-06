#pragma once

#include "Utilities/MPL/Algorithm.h"

#include <tuple>
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
        using type = typename std::tuple< Elements..., LastElement >;
    };

    template< typename Element, typename... Elements >
    using push_back_t = typename push_back< Element, Elements... >::type;

    template< typename Element, typename... Elements >
    struct push_front;

    template< typename Element, typename... Elements >
    struct push_front< Element, std::tuple< Elements... > > {
        using type = typename std::tuple< Element, Elements... >;
    };

    template< typename Element, typename... Elements >
    using push_front_t = typename push_front< Element, Elements... >::type;

    template< typename Tuple >
    using back_t = std::tuple_element_t< std::tuple_size< Tuple >::value - 1, Tuple >;

    template< typename Tuple >
    using front_t = std::tuple_element_t< 0, Tuple >;
} // namespace utils
