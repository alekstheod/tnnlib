#pragma once

#include <array>
#include <cstddef>
#include <tuple>

namespace nn::bp {

template<typename Var, std::size_t LayerSize, std::size_t InputsPerNeuron>
struct BPLayerContext {
    using VarType = Var;
    static constexpr auto layerSize = LayerSize;
    static constexpr auto inputsPerNeuron = InputsPerNeuron;

    std::array<Var, LayerSize> deltas{};
    std::array<Var, LayerSize> biasGradients{};
    std::array<std::array<Var, LayerSize>, InputsPerNeuron> weightGradients{};
};

namespace detail {

template<typename... LayerConfigs>
struct BPContext;

template<typename Var, typename... LayerConfigs>
struct BPContext<Var, LayerConfigs...> {
    using VarType = Var;

    template<std::size_t LayerIdx>
    auto& layerContext() {
        return std::get<LayerIdx>(m_layerContexts);
    }

    template<std::size_t LayerIdx>
    const auto& layerContext() const {
        return std::get<LayerIdx>(m_layerContexts);
    }

    void reset() {
        reset_impl(*this);
    }

private:
    template<typename Self>
    static void reset_impl(Self& self) {}

    std::tuple<LayerConfigs...> m_layerContexts{};
};

template<typename Var, typename... LayerConfigs>
auto makeBPContext() {
    return BPContext<Var, LayerConfigs...>{};
}

} // namespace detail

} // namespace nn::bp
