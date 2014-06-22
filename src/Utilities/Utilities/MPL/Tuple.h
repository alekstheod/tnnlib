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

template< template<class> class C, typename Tuple, unsigned int index>
struct rebind_tuple{
  typedef C<typename std::tuple_element<index, Tuple>::type> cur_type;
  typedef typename rebind_tuple<C, Tuple, index-1>::type next_type;
  cur_type* first = nullptr;
  next_type* second = nullptr;
  typedef decltype( std::tuple_cat( *second , std::tuple<cur_type>(*first)) ) type;
};

template< template<class> class C, typename Tuple>
struct rebind_tuple<C, Tuple, 0>{
  typedef std::tuple< C< typename std::tuple_element<0, Tuple>::type > > type;
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
struct rebind_tuple{
  typedef typename detail::rebind_tuple<C, Tuple, std::tuple_size<Tuple>::value - 1 >::type type;
};

}

#endif
