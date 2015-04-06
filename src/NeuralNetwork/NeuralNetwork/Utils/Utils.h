#ifndef NN_UTILS_H
#define NN_UTILS_H
#include <Utilities/MPL/Tuple.h>

namespace nn {

namespace detail {

/// workaround for VS++ compilation.
template <typename Var, typename T>
struct rebindOne {
    typedef typename T::template rebindVar<Var>::type type;
};

template<typename Var, typename T>
struct rebindVar;

template<typename Var, typename... T>
struct rebindVar<Var, std::tuple<T...> > {
    typedef typename std::tuple< typename rebindOne<Var, T>::type... > type;
};

template<std::size_t FirstInputs, typename RebindedTuple, typename Tuple>
struct RebindInputsHelper;

template<std::size_t FirstInputs, typename RebindedTuple, typename FirstLayer, typename... Layers>
struct RebindInputsHelper<FirstInputs, RebindedTuple, std::tuple<FirstLayer, Layers...>>
{                           
    typedef typename utils::push_back<typename FirstLayer::template rebindInputs<FirstInputs>::type, RebindedTuple>::type CurrentRebindedTuple;            
    typedef typename std::conditional<sizeof...(Layers) == 0,
	    CurrentRebindedTuple,
	    typename RebindInputsHelper<FirstLayer::CONST_NEURONS_NUMBER, CurrentRebindedTuple, std::tuple<Layers...>>::type>::type type;
};

template<std::size_t FirstInputs, typename RebindedTuple, typename... Layers>
struct RebindInputsHelper<FirstInputs, RebindedTuple, std::tuple<Layers...>>
{
    typedef RebindedTuple type;
};

template<std::size_t FirstInputs, typename Tuple>
struct rebindInputs
{
    static_assert(std::tuple_size<Tuple>::value >= 1, "Invalid");
    typedef typename RebindInputsHelper<FirstInputs, std::tuple<>, Tuple>::type type;
};

}

}

#endif
