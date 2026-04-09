#pragma once

#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"

#include <range/v3/view.hpp>
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

        using Memento = typename Base::Memento;
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
        using Base::getMemento;
        using Base::setMemento;

        BPNeuralLayer() {
            initializeBPState();
        }

        template< typename... Args >
        BPNeuralLayer(Args&&... args) : Base(std::forward< Args >(args)...) {
            initializeBPState();
        }

      private:
        void initializeBPState() {
            m_deltas.resize(size(), Var{});
            m_accumulatedWeightGradients.resize(size());
            m_accumulatedBiasGradient.resize(size(), Var{});

            for(std::size_t i = 0; i < size(); ++i) {
                m_accumulatedWeightGradients[i].resize(Grid::K::size, Var{});
            }
        }

      public:
        const Var& getDelta(std::size_t neuronId) const {
            return m_deltas[neuronId];
        }

        void setDelta(std::size_t neuronId, const Var& delta) {
            m_deltas[neuronId] = delta;
        }

        template< typename Prototype, typename MomentumFunc >
        void calculateDeltas(const Prototype& prototype, MomentumFunc momentum) {
            auto& outputFunc = std::get< 0 >(m_activationFunctions);
            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                auto& neuron = (*this)[neuronId];
                auto delta =
                 momentum(m_deltas[neuronId],
                          outputFunc.delta(neuron.getOutput(),
                                           std::get< 1 >(prototype)[neuronId]));
                m_deltas[neuronId] = delta;
            }
        }

        template< typename Optimizer >
        void calculateWeights(Optimizer& optimizer) {
            auto& self = *this;
            for(const auto neuronId : ranges::views::indices(size())) {
                auto& neuron = self[neuronId];
                const Var neuronDelta = m_deltas[neuronId];

                for(const auto weightId : ranges::views::indices(Grid::K::size)) {
                    const Var inputValue = neuron[weightId].value;
                    const Var weightGradient = neuronDelta * inputValue;
                    auto newWeight = optimizer(neuron[weightId].weight, weightGradient);
                    neuron[weightId].weight = newWeight;
                }

                Var bias = neuron.getBias();
                Var newBias = optimizer(bias, neuronDelta);
                neuron.setBias(newBias);
            }
        }

        void accumulateGradients() {
            auto& self = *this;
            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                auto& neuron = self[neuronId];
                const Var delta = m_deltas[neuronId];

                for(std::size_t weightId = 0; weightId < Grid::K::size; ++weightId) {
                    const Var inputValue = neuron[weightId].value;
                    m_accumulatedWeightGradients[neuronId][weightId] += inputValue * delta;
                }
                m_accumulatedBiasGradient[neuronId] += delta;
            }
        }

        template< typename Optimizer >
        void applyGradients(Optimizer& optimizer) {
            auto& self = *this;
            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                auto& neuron = self[neuronId];
                for(std::size_t weightId = 0; weightId < Grid::K::size; ++weightId) {
                    auto weight = neuron[weightId].weight;
                    auto gradient = m_accumulatedWeightGradients[neuronId][weightId];
                    auto newWeight = optimizer(weight, gradient);
                    neuron[weightId].weight = newWeight;
                }

                Var bias = neuron.getBias();
                Var gradient = m_accumulatedBiasGradient[neuronId];
                Var newBias = optimizer(bias, gradient);
                neuron.setBias(newBias);

                m_accumulatedWeightGradients[neuronId].assign(Grid::K::size, Var{});
                m_accumulatedBiasGradient[neuronId] = Var{};
            }
        }

        template< typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(AffectedLayer& affectedLayer, MomentumFunc momentum) {
            detail::calculateHiddenDeltas(*this, affectedLayer, momentum);
        }

        using Base::m_grid;

        std::vector< Var >& deltas() {
            return m_deltas;
        }
        const std::vector< Var >& deltas() const {
            return m_deltas;
        }

      private:
        std::vector< Var > m_deltas;
        std::vector< std::vector< Var > > m_accumulatedWeightGradients;
        std::vector< Var > m_accumulatedBiasGradient;
    };

} // namespace nn::bp
