#ifndef TYPELIST_H
#define TYPELIST_H
#include <Utilities/MPL/TypeTraits.h>

namespace utils{
  
struct NullType {};

template<class U, class T>
struct TypeList {
    typedef U Head;
    typedef T Tail;
};

template<class TList>
struct Length;

template<>
struct Length<NullType> {
    enum { value = 0 };
};

template<class T, class U>
struct Length< TypeList<T, U> > {
    enum { value = 1 + Length<U>::value };
};

template<>
struct Length< TypeList<NullType, NullType> > {
    enum { value = 0 };
};

template <class TList, class T> struct Append;

template <> struct Append<NullType, NullType> {
    typedef NullType Result;
};

template <class T> struct Append<NullType, T> {
    typedef TypeList<T,NullType> Result;
};

template <class Head, class Tail>
struct Append<NullType, TypeList<Head, Tail> > {
    typedef TypeList<Head, Tail> Result;
};

template <class Head, class Tail, class T>
struct Append< TypeList<Head, Tail >, T > {
    typedef TypeList<Head, typename Append<Tail, T>::Result> Result;
};

template <class TList> struct Reverse;

template <>
struct Reverse<NullType> {
    typedef NullType Result;
};

template <class Head, class Tail>
struct Reverse< TypeList<Head, Tail> > {
    typedef typename Append<typename Reverse<Tail>::Result, Head>::Result Result;
};

template<unsigned int position,class TList>
struct TypeAtPos {
    typedef typename TypeAtPos< position - 1, typename TList::Tail >::Result Result;
};

template<class Head, class Tail>
struct TypeAtPos<0, TypeList<Head,Tail> > {
    typedef Head Result;
};

template<unsigned int pos>
struct TypeAtPos<pos, NullType> {
    typedef NullType Result;
};

template<bool value, typename T1, typename T2>
struct if_;

template<typename T1, typename T2>
struct if_<true, T1, T2>{
  typedef T1 Result;
};

template<typename T1, typename T2>
struct if_<false, T1, T2>{
  typedef T2 Result;
};

template<typename T, typename TList>
struct IsInList{
  typedef typename IsInList< T, typename TList::Tail >::Result Result;
};

template<typename T>
struct IsInList< T, NullType >{
  typedef False Result;
};

template<typename T, typename U>
struct IsInList< T, TypeList<T, U> >{
  typedef True Result;
};

}

#define TYPELIST_0() utils::TypeList<utils::NullType, utils::NullType>
#define TYPELIST_1( Type ) utils::TypeList< Type, utils::NullType >
#define TYPELIST_2( Type1, Type2 ) utils::TypeList< Type1, TYPELIST_1( Type2 ) >
#define TYPELIST_3( Type1, Type2, Type3 ) utils::TypeList< Type1, TYPELIST_2( Type2, Type3 ) >
#define TYPELIST_4( Type1, Type2, Type3, Type4 ) utils::TypeList< Type1, TYPELIST_3( Type2, Type3, Type4 ) >
#define TYPELIST_5( Type1, Type2, Type3, Type4, Type5 ) utils::TypeList< Type1, TYPELIST_4( Type2, Type3, Type4, Type5 ) >
#define TYPELIST_6( Type1, Type2, Type3, Type4, Type5, Type6 ) utils::TypeList< Type1, TYPELIST_5( Type2, Type3, Type4, Type5, Type6 ) >

#endif
