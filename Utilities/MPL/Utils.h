#pragma once

namespace utils {
    template< typename T >
    T& deref(T* ptr) {
        return *ptr;
    }

    template< typename T >
    T& deref(T& value) {
        return value;
    }
} // namespace utils
