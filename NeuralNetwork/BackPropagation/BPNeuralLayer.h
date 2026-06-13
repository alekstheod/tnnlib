#pragma once

#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"

#include <MPL/TypeTraits.h>

#include <range/v3/all.hpp>
#include <array>

namespace nn::bp {
    template< typename Internal >
    struct BPNeuralLayer;

    namespace detail {

        template< typename BPCtx, std::size_t currentIdx, std::size_t affectedIdx, typename CurrentLayer, typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(CurrentLayer& currentLayer,
                                   BPCtx& ctx,
                                   AffectedLayer& affectedLayer,
                                   MomentumFunc momentum) {
            using Var = typename AffectedLayer::Var;
            auto& funcs = currentLayer.activationFunctions();
            auto& outputFunc = std::get< 0 >(funcs);
            auto& currentDeltas = std::get< currentIdx >(ctx.deltas);
            auto& currentOutputs = std::get< currentIdx >(ctx.outputs);
            auto& affectedDeltas = std::get< affectedIdx >(ctx.deltas);
            const auto& affectedWeights = std::get< affectedIdx >(ctx.weights);
            constexpr auto affectedInputs = affectedLayer.inputs();
            constexpr auto affectedSize = affectedLayer.size();

            currentLayer.for_each([&](auto i, auto&) {
                Var sum{};
                if (i.value < affectedInputs) {
                    for (std::size_t j = 0; j < affectedSize; ++j) {
                        sum += affectedDeltas[j] * affectedWeights[j * affectedInputs + i.value];
                    }
                }
                currentDeltas[i.value] =
                 momentum(currentDeltas[i.value],
                          sum * outputFunc.derivate(currentOutputs[i.value]));
            });
        }

    } // namespace detail

    template< typename NeuralLayerType >
    struct BPNeuralLayer : NeuralLayerType {
        using Base = NeuralLayerType;

        using NeuralLayer = NeuralLayerType;
        using Var = typename NeuralLayer::Var;
        using ActivationFunctions = typename NeuralLayer::ActivationFunctions;

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
        using adjust =
         BPNeuralLayer< typename NeuralLayerType::template adjust< inputs > >;

        using Base::for_each;
        using Base::inputs;
        using Base::size;
        using Base::operator[];

        BPNeuralLayer() = default;

        template< typename... Args >
        BPNeuralLayer(Args&&... args) : Base(std::forward< Args >(args)...) {
        }

        template< typename BPCtx, std::size_t myIdx, typename Prototype, typename MomentumFunc >
        void calculateDeltas(BPCtx& ctx, const Prototype& prototype, MomentumFunc momentum) {
            auto& outputFunc = std::get< 0 >(m_activationFunctions);
            auto& outputs = std::get< myIdx >(ctx.outputs);
            auto& deltas = std::get< myIdx >(ctx.deltas);

            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                deltas[neuronId] =
                 momentum(deltas[neuronId],
                          outputFunc.delta(outputs[neuronId],
                                           std::get< 1 >(prototype)[neuronId]));
            }
        }

        template< typename BPCtx, std::size_t myIdx, std::size_t affIdx, typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(BPCtx& ctx, AffectedLayer& affectedLayer, MomentumFunc momentum) {
            detail::calculateHiddenDeltas< BPCtx, myIdx, affIdx >(*this, ctx, affectedLayer, momentum);
        }

        template< typename BPCtx, std::size_t myIdx >
        void calculateWeights(BPCtx& ctx, const Var& learningRate) {
            auto& deltas = std::get< myIdx >(ctx.deltas);
            auto& weights = std::get< myIdx >(ctx.weights);
            auto& biases = std::get< myIdx >(ctx.biases);
            constexpr auto inputsNumber = inputs();

            if constexpr (myIdx > 0) {
                auto& predecessorOutputs = std::get< myIdx - 1 >(ctx.outputs);
                const auto inputSize = predecessorOutputs.size() < inputsNumber
                                        ? predecessorOutputs.size() : inputsNumber;
                for_each([&](auto i, auto&) {
                    auto delta = deltas[i.value];
                    for(std::size_t j = 0; j < inputSize; j++) {
                        auto input = predecessorOutputs[j];
                        auto weight = weights[i.value * inputsNumber + j];
                        weights[i.value * inputsNumber + j] = weight - learningRate * input * delta;
                    }
                    biases[i.value] = biases[i.value] - learningRate * delta;
                });
            } else {
                for_each([&](auto i, auto& neuron) {
                    auto delta = deltas[i.value];
                    for(std::size_t j = 0; j < inputsNumber; j++) {
                        auto input = neuron[j].value;
                        auto weight = weights[i.value * inputsNumber + j];
                        weights[i.value * inputsNumber + j] = weight - learningRate * input * delta;
                    }
                    biases[i.value] = biases[i.value] - learningRate * delta;
                });
            }
        }

        template< typename BPCtx, std::size_t myIdx >
        void accumulateGradients(BPCtx& ctx) {
            auto& deltas = std::get< myIdx >(ctx.deltas);
            auto& weightGrads = std::get< myIdx >(ctx.weightGradients);
            auto& biasGrads = std::get< myIdx >(ctx.biasGradients);
            constexpr auto inputsNumber = inputs();

            if constexpr (myIdx > 0) {
                auto& predecessorOutputs = std::get< myIdx - 1 >(ctx.outputs);
                const auto inputSize = predecessorOutputs.size() < inputsNumber
                                        ? predecessorOutputs.size() : inputsNumber;
                for_each([&](auto i, auto&) {
                    auto delta = deltas[i.value];
                    for(std::size_t j = 0; j < inputSize; j++) {
                        weightGrads[i.value * inputsNumber + j] += predecessorOutputs[j] * delta;
                    }
                    biasGrads[i.value] += delta;
                });
            } else {
                for_each([&](auto i, auto& neuron) {
                    auto delta = deltas[i.value];
                    for(std::size_t j = 0; j < inputsNumber; j++) {
                        auto input = neuron[j].value;
                        weightGrads[i.value * inputsNumber + j] += input * delta;
                    }
                    biasGrads[i.value] += delta;
                });
            }
        }

        template< typename BPCtx, std::size_t myIdx >
        void applyGradients(BPCtx& ctx, const Var& learningRate) {
            auto& weightGrads = std::get< myIdx >(ctx.weightGradients);
            auto& biasGrads = std::get< myIdx >(ctx.biasGradients);
            auto& weights = std::get< myIdx >(ctx.weights);
            auto& biases = std::get< myIdx >(ctx.biases);
            constexpr auto inputsNumber = inputs();

            for_each([&](auto i, auto&) {
                for(std::size_t j = 0; j < inputsNumber; j++) {
                    weights[i.value * inputsNumber + j] -=
                     learningRate * weightGrads[i.value * inputsNumber + j];
                    weightGrads[i.value * inputsNumber + j] = Var{};
                }

                biases[i.value] -= learningRate * biasGrads[i.value];
                biasGrads[i.value] = Var{};
            });
        }
    };
} // namespace nn::bp
