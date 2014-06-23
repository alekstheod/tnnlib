#ifndef ConfigH
#define ConfigH

#if defined(_MSC_VER)
# define NN_CC_MSVC
#elif defined(__clang__)
# define NN_CC_CLANG
#elif defined(__GNUC__)
# define NN_CC_GNU
#endif

#endif
