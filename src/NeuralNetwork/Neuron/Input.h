#pragma once

#include <System/Time.h>

namespace nn {
    template< typename Var >
    struct Input {
        Var weight = utils::createRandom< Var >(1) / Var{100.f};
        Var value = {};
    };
} // namespace nn
