#pragma once

#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"

#include <MPL/TypeTraits.h>

#include <range/v3/all.hpp>
#include <array>

namespace nn::bp {
    template< typename Internal >
    struct BPNeuralLayer;

    namespace detail {

        template< typename CurrentLayer, typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(CurrentLayer& currentLayer,
                                   AffectedLayer& affectedLayer,
                                   MomentumFunc momentum) {
            using Var = typename AffectedLayer::Var;
            auto& funcs = currentLayer.activationFunctions();
            auto& outputFunc = std::get< 0 >(funcs);

            currentLayer.for_each(
             [&currentLayer, &affectedLayer, &momentum, &outputFunc](auto i, auto& currentNeuron) {
                 Var sum{};
                 affectedLayer.for_each([&sum, &i, &affectedLayer](auto j, auto& neuron) {
                     auto affectedDelta = affectedLayer.getDelta(j.value);
                     auto affectedWeight = neuron.getWeight(i.value);
                     sum += affectedDelta * affectedWeight;
                 });

                 currentLayer.setDelta(i.value,
                                       momentum(currentLayer.getDelta(i.value),
                                                sum * outputFunc.derivate(
                                                       currentNeuron.getOutput())));
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

        BPNeuralLayer() {
            initializeBPState();
        }

        template< typename... Args >
        BPNeuralLayer(Args&&... args) : Base(std::forward< Args >(args)...) {
            initializeBPState();
        }

      private:
        void initializeBPState() {
            for(std::size_t i = 0; i < size(); ++i) {
                m_accumulatedWeightGradients[i].fill(Var{});
                m_accumulatedBiasGradient[i] = Var{};
            }
        }

      public:
        const Var& getDelta(std::size_t neuronId) const {
            return m_deltas[neuronId];
        }

        void setDelta(std::size_t neuronId, const Var& delta) {
            m_deltas[neuronId] = delta;
        }

        void setDelta(std::size_t neuronId, Var&& delta) {
            m_deltas[neuronId] = std::move(delta);
        }

        std::array< Var, NeuralLayerType::size() >& deltas() {
            return m_deltas;
        }

        const std::array< Var, NeuralLayerType::size() >& deltas() const {
            return m_deltas;
        }

        /**
         * @brief Will calculate the deltas for the current layer. This
         * method must be called for the output layer layers.
         * @param current data set based on which the deltas will be
         * calculated.
         */
        template< typename Prototype, typename MomentumFunc >
        void calculateDeltas(const Prototype& prototype, MomentumFunc momentum) {
            auto& outputFunc = std::get< 0 >(m_activationFunctions);
            std::size_t neuronId = 0;
            for(auto& neuron : *this) {
                auto delta =
                 momentum(m_deltas[neuronId],
                          outputFunc.delta(neuron.getOutput(),
                                           std::get< 1 >(prototype)[neuronId]));
                m_deltas[neuronId] = delta;
                neuronId++;
            }
        }

        template< typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(AffectedLayer& affectedLayer, MomentumFunc momentum) {
            detail::calculateHiddenDeltas(*this, affectedLayer, momentum);
        }

        template< typename Optimizer >
        void calculateWeights(Optimizer& optimizer) {
            for_each([this, &optimizer](auto i, auto& neuron) {
                std::size_t inputsNumber = neuron.size();
                auto delta = m_deltas[i.value];
                for(std::size_t j = 0; j < inputsNumber; j++) {

                    auto input = neuron[j].value;
                    auto weight = neuron[j].weight;
                    auto gradient = input * delta;
                    auto newWeight = optimizer(weight, gradient);
                    neuron.setWeight(j, newWeight);
                }

                Var weight = neuron.getBias();
                auto gradient = delta;
                auto newBias = optimizer(weight, gradient);
                neuron.setBias(newBias);
            });
        }

        void accumulateGradients() {
            for_each([this](auto i, auto& neuron) {
                const auto inputsNumber = neuron.size();
                auto delta = m_deltas[i.value];
                for(std::size_t j = 0; j < inputsNumber; j++) {
                    auto input = neuron[j].value;
                    m_accumulatedWeightGradients[i.value][j] += input * delta;
                }
                m_accumulatedBiasGradient[i.value] += delta;
            });
        }

        template< typename Optimizer >
        void applyGradients(Optimizer& optimizer) {
            for_each([this, &optimizer](auto i, auto& neuron) {
                std::size_t inputsNumber = neuron.size();
                for(std::size_t j = 0; j < inputsNumber; j++) {
                    auto weight = neuron[j].weight;
                    auto gradient = m_accumulatedWeightGradients[i.value][j];
                    auto newWeight = optimizer(weight, gradient);
                    neuron.setWeight(j, newWeight);
                }

                Var weight = neuron.getBias();
                Var gradient = m_accumulatedBiasGradient[i.value];
                auto newBias = optimizer(weight, gradient);
                neuron.setBias(newBias);

                m_accumulatedWeightGradients[i.value].assign(inputsNumber, Var{});
                m_accumulatedBiasGradient[i.value] = Var{};
            });
        }

        Var getAccumulatedGradient(std::size_t neuronId, std::size_t inputIdx) const {
            return m_accumulatedWeightGradients[neuronId][inputIdx];
        }

        Var getAccumulatedBiasGradient(std::size_t neuronId) const {
            return m_accumulatedBiasGradient[neuronId];
        }

        void resetGradients() {
            for(std::size_t i = 0; i < size(); ++i) {
                m_accumulatedWeightGradients[i].fill(Var{});
                m_accumulatedBiasGradient[i] = Var{};
            }
        }

      private:
        std::array< Var, size() > m_deltas{};
        std::array< std::array< Var, inputs() >, size() > m_accumulatedWeightGradients{};
        std::array< Var, size() > m_accumulatedBiasGradient{};
    };
} // namespace nn::bp
