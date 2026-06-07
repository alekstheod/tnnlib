#pragma once

#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"

#include <range/v3/view.hpp>
#include <algorithm>
#include <vector>

namespace nn::bp {
    template< typename >
    struct BPNeuralLayer;

    template< typename LayerType, typename Grid >
    struct BPNeuralLayer< nn::detail::ConvolutionLayer< LayerType, Grid > >
     : private nn::detail::ConvolutionLayer< LayerType, Grid > {
        using Base = nn::detail::ConvolutionLayer< LayerType, Grid >;

        using NeuralLayerType = typename nn::detail::ConvolutionLayer< LayerType, Grid >;

        using Var = typename NeuralLayerType::Var;
        using ActivationFunctions = typename NeuralLayerType::ActivationFunctions;

      private:
        ActivationFunctions m_activationFunctions{};

      public:
        ActivationFunctions& activationFunctions() {
            return m_activationFunctions;
        }

        const ActivationFunctions& activationFunctions() const {
            return m_activationFunctions;
        }

        template< typename VarType >
        using use = BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

        template< std::size_t inputs >
        using adjust = BPNeuralLayer;

        using Base::begin;
        using Base::cbegin;
        using Base::cend;
        using Base::end;
        using Base::for_each;
        using Base::inputs;
        using Base::setInput;
        using Base::size;
        using Base::operator[];
        using Base::calculateOutputs;

        BPNeuralLayer() = default;

        template< typename... Args >
        BPNeuralLayer(Args&&... args) : Base(std::forward< Args >(args)...) {}

        template< typename BPCtx, std::size_t myIdx, typename Prototype, typename MomentumFunc >
        void calculateDeltas(BPCtx& ctx, const Prototype& prototype, MomentumFunc momentum) {
            auto& outputFunc = std::get< 0 >(m_activationFunctions);
            auto& outputs = std::get< myIdx >(ctx.outputs);
            auto& deltas = std::get< myIdx >(ctx.deltas);
            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                deltas[neuronId] = momentum(deltas[neuronId],
                                            outputFunc.delta(outputs[neuronId],
                                                             std::get< 1 >(prototype)[neuronId]));
            }
        }

        template< typename BPCtx, std::size_t myIdx >
        void calculateWeights(BPCtx& ctx, const Var& learningRate) {
            auto& deltas = std::get< myIdx >(ctx.deltas);
            auto& weights = std::get< myIdx >(ctx.weights);
            auto& biases = std::get< myIdx >(ctx.biases);
            auto& self = *this;
            for(const auto neuronId : ranges::views::indices(size())) {
                auto& neuron = self[neuronId];
                const Var neuronDelta = deltas[neuronId];

                for(const auto weightId : ranges::views::indices(Grid::K::size)) {
                    const Var inputValue = neuron[weightId].value;
                    const Var weightGradient = neuronDelta * inputValue;
                    weights[neuronId * Grid::K::size + weightId] -= learningRate * weightGradient;
                }

                biases[neuronId] -= learningRate * neuronDelta;
            }
        }

        template< typename BPCtx, std::size_t myIdx >
        void accumulateGradients(BPCtx& ctx) {
            auto& deltas = std::get< myIdx >(ctx.deltas);
            auto& weightGrads = std::get< myIdx >(ctx.weightGradients);
            auto& biasGrads = std::get< myIdx >(ctx.biasGradients);
            auto& self = *this;
            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                auto& neuron = self[neuronId];
                const Var delta = deltas[neuronId];

                for(std::size_t weightId = 0; weightId < Grid::K::size; ++weightId) {
                    const Var inputValue = neuron[weightId].value;
                    weightGrads[neuronId * Grid::K::size + weightId] += inputValue * delta;
                }
                biasGrads[neuronId] += delta;
            }
        }

        template< typename BPCtx, std::size_t myIdx >
        void applyGradients(BPCtx& ctx, const Var& learningRate) {
            auto& weightGrads = std::get< myIdx >(ctx.weightGradients);
            auto& biasGrads = std::get< myIdx >(ctx.biasGradients);
            auto& weights = std::get< myIdx >(ctx.weights);
            auto& biases = std::get< myIdx >(ctx.biases);
            auto& self = *this;
            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                for(std::size_t weightId = 0; weightId < Grid::K::size; ++weightId) {
                    weights[neuronId * Grid::K::size + weightId] -=
                     learningRate * weightGrads[neuronId * Grid::K::size + weightId];
                    weightGrads[neuronId * Grid::K::size + weightId] = Var{};
                }

                biases[neuronId] -= learningRate * biasGrads[neuronId];
                biasGrads[neuronId] = Var{};
            }
        }

        template< typename BPCtx, std::size_t myIdx, std::size_t affIdx, typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(BPCtx& ctx, AffectedLayer& affectedLayer, MomentumFunc momentum) {
            detail::calculateHiddenDeltas< BPCtx, myIdx, affIdx >(*this, ctx, affectedLayer, momentum);
        }

        using Base::m_grid;
    };

} // namespace nn::bp