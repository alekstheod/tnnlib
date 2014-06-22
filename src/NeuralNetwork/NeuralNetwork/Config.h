#ifndef ConfigH
#define ConfigH

#if defined(_MSC_VER)
# define NN_CC_MSVC
#elif defined(__clang__)
# define NN_CC_CLANG
#elif defined(__GNUC__)
# define NN_CC_GNU
#endif

#if defined(NN_CC_MSVC)
# define NN_CONSTEXPR
#else
# define NN_CONSTEXPR constexpr
#endif

#if defined(NN_CC_MSVC)
# define NN_DEFINE_CONST(TYPE, NAME, VALUE) enum { NAME = VALUE }
#else
# define NN_DEFINE_CONST(TYPE, NAME, VALUE) NN_CONSTEXPR static TYPE NAME = VALUE
#endif

#endif
