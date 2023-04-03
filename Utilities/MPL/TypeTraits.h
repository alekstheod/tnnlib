#pragma once

#include <type_traits>

namespace utils {

    template< typename T >
    struct hasArrowOperator {
        template< typename C >
        static char hasArrow(decltype(&C::operator->));

        template< typename C >
        static double hasArrow(...);

        enum { value = sizeof(hasArrow< T >(0)) == sizeof(char) };
    };

    struct empty {};

    template< typename T, typename T2 >
    struct is_same_template : std::false_type {};

    template< template< class > class T, typename T2 >
    struct is_same_template< T< T2 >, T< T2 > > : std::true_type {};

    template< typename T, typename T2 >
    constexpr auto is_same_template_v = is_same_template< T, T2 >::value();
} // namespace utils
