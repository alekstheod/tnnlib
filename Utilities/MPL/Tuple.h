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
    struct pop_back;

    template< std::size_t Start, std::size_t End, typename... Elements >
    struct tuple_element_sequence {
        static_assert(Start <= End && End <= sizeof...(Elements),
                      "Invalid range");
        using type = decltype(std::tuple_cat(
         std::declval< std::tuple< std::tuple_element_t< Start, std::tuple< Elements... > > > >(),
         typename tuple_element_sequence< Start + 1, End, Elements... >::type{}));
    };

    template< std::size_t N, typename... Elements >
    struct tuple_element_sequence< N, N, Elements... > {
        using type = std::tuple<>;
    };

    template< std::size_t Start, std::size_t End, typename... Elements >
    using tuple_element_sequence_t =
     typename tuple_element_sequence< Start, End, Elements... >::type;

    template< typename... Elements >
    struct pop_back< std::tuple< Elements... > > {
        static constexpr auto size = sizeof...(Elements);
        using type = tuple_element_sequence_t< 0, size - 1, Elements... >;
    };

    template< typename Tuple >
    using pop_back_t = typename pop_back< Tuple >::type;

    template< typename Tuple >
    using back_t = std::tuple_element_t< std::tuple_size< Tuple >::value - 1, Tuple >;

    template< typename Tuple >
    using front_t = std::tuple_element_t< 0, Tuple >;
} // namespace utils
