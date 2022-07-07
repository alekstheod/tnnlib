#pragma once
#include <utility>
#include <tuple>
#include <vector>

namespace utils {

    template< std::size_t N >
    struct num {
        static const constexpr auto value = N;
    };

    template< class F, std::size_t... Is >
    void for_(F func, std::index_sequence< Is... >) {
        using expander = int[];
        (void)expander{0, ((void)func(num< Is >{}), 0)...};
    }

    template< std::size_t N, typename F >
    void for_(F func) {
        for_(func, std::make_index_sequence< N >());
    }

    template< std::size_t idx, typename... T >
    auto& get(std::tuple< T... >& container) {
        return std::get< idx >(container);
    }

    template< std::size_t idx, typename... T >
    const auto& get(const std::tuple< T... >& container) {
        return std::get< idx >(container);
    }

    template< std::size_t idx, typename T >
    auto& get(T& container) {
        return container[idx];
    }

    template< std::size_t idx, typename T >
    const auto& get(const T& container) {
        return container[idx];
    }

    template< typename... T >
    constexpr auto size_of(const std::tuple< T... >&) {
        return sizeof...(T);
    }

    template< typename T, std::size_t size >
    constexpr auto size_of(const std::array< T, size >&) {
        return size;
    }
} // namespace utils
