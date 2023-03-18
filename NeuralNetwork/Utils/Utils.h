#pragma once
#include <MPL/Tuple.h>

namespace nn {

    namespace detail {
        namespace mpl {
            template< std::size_t FirstInputs, typename RebindedTuple, typename Tuple >
            struct RebindInputsHelper;

            template< std::size_t FirstInputs, typename RebindedTuple, typename FirstLayer, typename... Layers >
            struct RebindInputsHelper< FirstInputs, RebindedTuple, std::tuple< FirstLayer, Layers... > > {
                typedef
                 typename utils::push_back< typename FirstLayer::template adjust< FirstInputs >, RebindedTuple >::type CurrentRebindedTuple;
                typedef typename std::conditional<
                 sizeof...(Layers) == 0,
                 CurrentRebindedTuple,
                 typename RebindInputsHelper< FirstLayer::size(), CurrentRebindedTuple, std::tuple< Layers... > >::type >::type type;
            };

            template< std::size_t FirstInputs, typename RebindedTuple, typename... Layers >
            struct RebindInputsHelper< FirstInputs, RebindedTuple, std::tuple< Layers... > > {
                typedef RebindedTuple type;
            };

            template< std::size_t FirstInputs, typename Tuple >
            struct rebindInputs {
                static_assert(std::tuple_size< Tuple >::value >= 1, "Invalid");
                typedef
                 typename RebindInputsHelper< FirstInputs, std::tuple<>, Tuple >::type type;
            };
        } // namespace mpl
    } // namespace detail
} // namespace nn
