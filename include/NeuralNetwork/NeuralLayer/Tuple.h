#pragma once

#include <NeuralNetwork/NeuralLayer/Container.h>

#include <tuple>

namespace nn {
    namespace detail {
        template< typename... T >
        struct Layer< std::tuple< T... > > {
            using Container = std::tuple< T... >;

            Container m_neurons;
        };
    } // namespace detail
} // namespace nn
