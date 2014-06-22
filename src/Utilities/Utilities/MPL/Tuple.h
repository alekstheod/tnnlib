#ifndef Tuple_h
#define Tuple_h
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace utils
{

namespace detail
{

template < uint N >
struct for_each_t {
    template < typename Functor, typename... ArgsT >
    static void exec ( std::tuple<ArgsT...>& t, Functor func ) {
        for_each_t<N-1>::exec ( t, func );
	func ( std::get<N-1> ( t ) );
    }
};

template <>
struct for_each_t<0> {
    template < typename Functor, typename... ArgsT>
    static void exec ( std::tuple<ArgsT...>& t, Functor func ) {
    }
};

template < uint N >
struct for_each_t_c {
    template < typename Functor, typename... ArgsT >
    static void exec ( const std::tuple<ArgsT...>& t, Functor func ) {
        for_each_t_c<N-1>::exec ( t, func );
	func ( std::get<N-1> ( t ) );
    }
};

template <>
struct for_each_t_c<0> {
    template < typename Functor, typename... ArgsT>
    static void exec ( const std::tuple<ArgsT...>& t, Functor func ) {
    }
};


template <class T, std::size_t N, class... Args>
struct get
{
    static const auto value = N;
};

template <class T, std::size_t N, class... Args>
struct get<T, N, T, Args...>
{
    static const auto value = N;
};

template <class T, std::size_t N, class U, class... Args>
struct get<T, N, U, Args...>
{
    static const auto value = get<T, N + 1, Args...>::value;
};

} // namespace detail


template <class T, class... Args>
T& get(std::tuple<Args...>& t)
{
    return std::get<detail::get<T, 0, Args...>::value>(t);
}

template < typename Functor, typename... ArgsT >
void for_each( std::tuple<ArgsT...>& t, Functor func )
{
    detail::for_each_t<sizeof...(ArgsT)>::exec( t, func );
}

template < typename Functor, typename... ArgsT >
void for_each_c( const std::tuple<ArgsT...>& t, Functor func )
{
    detail::for_each_t_c<sizeof...(ArgsT)>::exec( t, func );
}

template<template<typename> class E, typename Tuple>
struct rebind_tuple;

template<template<typename> class E, typename... Args>
struct rebind_tuple<E, std::tuple<Args...>>
{
   typedef std::tuple<E<Args>...> type;
};

}

#endif
