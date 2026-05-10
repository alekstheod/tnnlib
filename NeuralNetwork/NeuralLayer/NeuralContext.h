#pragma once

#include <array>
#include <cstddef>
#include <tuple>

namespace nn {

template<typename Var, std::size_t LayerSize, std::size_t InputsPerNeuron>
struct LayerContext {
    using VarType = Var;
    static constexpr auto layerSize = LayerSize;
    static constexpr auto inputsPerNeuron = InputsPerNeuron;

    std::array<Var, LayerSize> outputs{};
    std::array<std::array<Var, InputsPerNeuron>, LayerSize> inputs{};
};

namespace detail {

template<typename... LayerContexts>
struct NeuralContext;

template<typename Var>
struct NeuralContext<Var> {
    using VarType = Var;

    void reset() {}

    static constexpr std::size_t layerCount = 0;

    template<std::size_t LayerIdx>
    auto& layerContext() {
        static_assert(LayerIdx < layerCount, "Layer index out of bounds");
        return *static_cast<Head*>(nullptr);
    }

    template<std::size_t LayerIdx>
    const auto& layerContext() const {
        static_assert(LayerIdx < layerCount, "Layer index out of bounds");
        return *static_cast<const Head*>(nullptr);
    }

private:
    using Head = void;
};

template<typename Var, typename Head, typename... Tail>
struct NeuralContext<Var, Head, Tail...> {
    using VarType = Var;

    void reset() {
        head.outputs = {};
        for (auto& arr : head.inputs) {
            arr.fill(Var{});
        }
        tail.reset();
    }

    static constexpr std::size_t layerCount = 1 + sizeof...(Tail);

    Head head;
    NeuralContext<Var, Tail...> tail;

    template<std::size_t LayerIdx>
    auto& layerContext() {
        return layerContext_impl<LayerIdx>(*this);
    }

    template<std::size_t LayerIdx>
    const auto& layerContext() const {
        return layerContext_impl<LayerIdx>(*this);
    }

private:
    template<std::size_t N>
    static auto& layerContext_impl(NeuralContext<Var, Head, Tail...>& ctx) {
        if constexpr (N == 0) {
            return ctx.head;
        } else {
            return ctx.tail.template layerContext<N - 1>();
        }
    }

    template<std::size_t N>
    static const auto& layerContext_impl(const NeuralContext<Var, Head, Tail...>& ctx) {
        if constexpr (N == 0) {
            return ctx.head;
        } else {
            return ctx.tail.template layerContext<N - 1>();
        }
    }
};

template<typename Var, typename... LayerContexts>
auto makeNeuralContext() {
    return NeuralContext<Var, LayerContexts...>{};
}

} // namespace detail

} // namespace nn
