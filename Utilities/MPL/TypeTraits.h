#ifndef TYPETRAITS_h
#define TYPETRAITS_h
#include <tuple>
#include "Tuple.h"

namespace utils
{

template <typename T>
struct hasArrowOperator
{
    template <typename C>
    static char hasArrow(decltype(&C::operator->));

    template <typename C>
    static double hasArrow(...);

    enum
    {
        value = sizeof(hasArrow<T>(0)) == sizeof(char)
    };
};

struct empty
{
};
} // namespace utils

#endif
