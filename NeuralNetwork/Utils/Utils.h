#pragma once
#include <MPL/Tuple.h>

namespace nn::detail::mpl {

    template< typename... Layers >
    struct adjustInputs;

    template< typename Pred, typename Suc >
    struct adjustInputs< Pred, Suc > {
        using type = std::tuple< typename Suc::template adjust< Pred::size() > >;
    };

    template< typename Pred, typename Suc, typename... Layers >
    struct adjustInputs< Pred, Suc, Layers... > {
        using type =
         utils::push_front_t< typename Suc::template adjust< Pred::size() >,
                              typename adjustInputs< Suc, Layers... >::type >;
    };

    template< typename... Layers >
    struct rebindInputs;

    template< typename FirstLayer, typename... Layers >
    struct rebindInputs< std::tuple< FirstLayer, Layers... > > {
        using type =
         utils::push_front_t< FirstLayer, typename adjustInputs< FirstLayer, Layers... >::type >;
    };
} // namespace nn::detail::mpl
