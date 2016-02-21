#ifndef Tuple_h
#define Tuple_h
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace utils
{
  template<typename LastElement, typename... Elements>
  struct push_back;
  
  template<typename Tuple>
  struct reverse;
  
  namespace detail
  {

    template < unsigned int N >
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

    template < unsigned int N >
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

    template<template<typename> class C, typename Tuple>
    struct rebind_tuple;

    template<template<typename> class C, typename... Args>
    struct rebind_tuple<C, std::tuple<Args...>>
    {
      typedef typename std::tuple<C<Args>...> type;
    };
    
    template<typename... Next>
    struct reverse;
    
    template<typename First>
    struct reverse<First>{
      using type = std::tuple<First>;
    };
    
    template<typename First, typename... Args>
    struct reverse<First, Args...>{
      using type = typename push_back<First, typename reverse<Args...>::type>::type;
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

  template< template<class> class C, typename Tuple>
  struct rebind_tuple
  {
    typedef typename detail::rebind_tuple<C, Tuple>::type type;
  };

  template<typename LastElement, typename... Elements>
  struct push_back<LastElement, std::tuple<Elements...> >
  {
      typedef typename std::tuple<Elements..., LastElement> type;
  };
  
  template<typename... Args>
  struct reverse<std::tuple<Args...>>{
    using type = typename detail::reverse<Args...>::type;
  };

}

#endif
