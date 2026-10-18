#pragma once

#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"

#include <MPL/TypeTraits.h>

#include <range/v3/all.hpp>
#include <array>

namespace nn::bp {
    template< typename Internal >
    struct BPNeuralLayer;

    namespace detail {

        /// Computes the deltas of layer `currentIdx` by collecting, for every
        /// neuron of that layer, the contributions of all later layers that
        /// read its outputs through the connection matrix. Connections from
        /// the same or earlier layers read values from a previous evaluation
        /// and are treated as constants (no gradient flows through them).
        template< typename BPCtx, std::size_t currentIdx, typename CurrentLayer, typename MomentumFunc >
        void calculateHiddenDeltas(CurrentLayer& currentLayer,
                                   BPCtx& ctx,
                                   MomentumFunc momentum) {
            using Var = typename CurrentLayer::Var;
            auto& funcs = currentLayer.activationFunctions();
            auto& outputFunc = std::get< 0 >(funcs);
            auto& currentDeltas = std::get< currentIdx >(ctx.deltas);
            auto& currentOutputs = std::get< currentIdx >(ctx.outputs);
            constexpr auto numLayers = std::tuple_size< std::decay_t< decltype(ctx.deltas) > >::value;
            constexpr auto currentSize = CurrentLayer::size();

            std::array< Var, currentSize > sums{};
            utils::for_< numLayers >([&](auto src) {
                constexpr auto srcIdx = src.value;
                if constexpr (srcIdx > currentIdx) {
                    const auto& srcConnections = std::get< srcIdx >(ctx.connections);
                    const auto& srcDeltas = std::get< srcIdx >(ctx.deltas);
                    const auto& srcWeights = std::get< srcIdx >(ctx.weights);
                    const auto srcInputs = srcWeights.size() / srcDeltas.size();
                    for(std::size_t j = 0; j < srcDeltas.size(); ++j) {
                        for(std::size_t s = 0; s < srcConnections[j].size(); ++s) {
                            const auto& conn = srcConnections[j][s];
                            if(conn.layer == currentIdx && conn.neuron < currentSize) {
                                sums[conn.neuron] += srcDeltas[j] * srcWeights[j * srcInputs + s];
                            }
                        }
                    }
                }
            });

            currentLayer.for_each([&](auto i, auto&) {
                currentDeltas[i.value] =
                 momentum(currentDeltas[i.value],
                          sums[i.value] * outputFunc.derivate(currentOutputs[i.value]));
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

        template< typename BPCtx, std::size_t myIdx, typename MomentumFunc >
        void calculateHiddenDeltas(BPCtx& ctx, MomentumFunc momentum) {
            detail::calculateHiddenDeltas< BPCtx, myIdx >(*this, ctx, momentum);
        }

        template< typename BPCtx, std::size_t myIdx >
        void calculateWeights(BPCtx& ctx, const Var& learningRate) {
            auto& deltas = std::get< myIdx >(ctx.deltas);
            auto& weights = std::get< myIdx >(ctx.weights);
            auto& biases = std::get< myIdx >(ctx.biases);
            const auto& connections = std::get< myIdx >(ctx.connections);
            constexpr auto inputsNumber = inputs();

            if constexpr (myIdx > 0) {
                const auto& outputs = ctx.outputs;
                for_each([&](auto i, auto&) {
                    const Var delta = deltas[i.value];
                    const auto& neuronConn = connections[i.value];
                    for(std::size_t s = 0; s < inputsNumber; ++s) {
                        if(const Var* input = outputOf< Var >(outputs, neuronConn[s])) {
                            weights[i.value * inputsNumber + s] -= learningRate * (*input) * delta;
                        }
                    }
                    biases[i.value] -= learningRate * delta;
                });
            } else {
                for_each([&](auto i, auto& neuron) {
                    const Var delta = deltas[i.value];
                    for(std::size_t s = 0; s < inputsNumber; ++s) {
                        const Var input = neuron[s].value;
                        weights[i.value * inputsNumber + s] -= learningRate * input * delta;
                    }
                    biases[i.value] -= learningRate * delta;
                });
            }
        }

        template< typename BPCtx, std::size_t myIdx >
        void accumulateGradients(BPCtx& ctx) {
            auto& deltas = std::get< myIdx >(ctx.deltas);
            auto& weightGrads = std::get< myIdx >(ctx.weightGradients);
            auto& biasGrads = std::get< myIdx >(ctx.biasGradients);
            const auto& connections = std::get< myIdx >(ctx.connections);
            constexpr auto inputsNumber = inputs();

            if constexpr (myIdx > 0) {
                const auto& outputs = ctx.outputs;
                for_each([&](auto i, auto&) {
                    const Var delta = deltas[i.value];
                    const auto& neuronConn = connections[i.value];
                    for(std::size_t s = 0; s < inputsNumber; ++s) {
                        if(const Var* input = outputOf< Var >(outputs, neuronConn[s])) {
                            weightGrads[i.value * inputsNumber + s] += (*input) * delta;
                        }
                    }
                    biasGrads[i.value] += delta;
                });
            } else {
                for_each([&](auto i, auto& neuron) {
                    const Var delta = deltas[i.value];
                    for(std::size_t s = 0; s < inputsNumber; ++s) {
                        const Var input = neuron[s].value;
                        weightGrads[i.value * inputsNumber + s] += input * delta;
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
                for(std::size_t s = 0; s < inputsNumber; ++s) {
                    weights[i.value * inputsNumber + s] -=
                     learningRate * weightGrads[i.value * inputsNumber + s];
                    weightGrads[i.value * inputsNumber + s] = Var{};
                }

                biases[i.value] -= learningRate * biasGrads[i.value];
                biasGrads[i.value] = Var{};
            });
        }
    };
} // namespace nn::bp
