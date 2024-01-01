#pragma once

#include <MPL/Tuple.h>

namespace nn::detail::mpl {

    template< typename Pred, typename Layer, typename... Other >
    constexpr auto adjust(Pred, Layer);

    template< typename Pred >
    constexpr auto adjust(Pred) -> Pred;

    // clang-format off
    template< typename Pred, typename Layer, typename... Other >
    constexpr auto adjust(Pred, Layer) 
		-> utils::push_back_t< typename Layer::template adjust< Pred::size() >,
							   decltype(adjust(std::declval< Layer >(), std::declval< Other >()...))
						  	 >;
    // clang-format on

    template< typename Pred, typename Layer, typename... Other >
    constexpr auto adjust(const std::tuple< Pred, Layer, Other... >&)
     -> utils::push_back_t< typename Layer::template adjust< Pred::size() >,
                            decltype(adjust(std::declval< Layer >(), std::declval< Other >()...)) >;

} // namespace nn::detail::mpl
