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
     : public nn::detail::ConvolutionLayer< LayerType, Grid > {

        using Base = nn::detail::ConvolutionLayer< LayerType, Grid >;
        using NeuralLayerType = Base;
        using Var = typename NeuralLayerType::Var;

        template< typename VarType >
        using use = BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

        template< std::size_t inputs >
        using adjust = BPNeuralLayer;

        static constexpr auto size() {
            return NeuralLayerType::size();
        }

        static constexpr auto inputs() {
            return NeuralLayerType::inputs();
        }

        using Memento = typename Base::Memento;

        using Base::begin;
        using Base::cbegin;
        using Base::cend;
        using Base::end;
        using Base::for_each;
        using Base::setInput;
        using Base::operator[];
        using Base::calculateOutputs;
        using Base::getMemento;
        using Base::setMemento;

        using Base::m_grid;


        template< typename Prototype, typename MomentumFunc >
        void calculateDeltas(const Prototype& prototype, MomentumFunc momentum) {
            auto& outputs = std::get< 1 >(prototype);
            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                auto& neuron = (*this)[neuronId];
                auto expectedOutput = outputs[neuronId];
                auto actualOutput = neuron.getOutput();
                auto delta =
                 momentum(m_deltas[neuronId],
                          neuron.getOutputFunction().delta(actualOutput, expectedOutput));
                m_deltas[neuronId] = delta;
            }
        }

        template< typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(AffectedLayer& affectedLayer, MomentumFunc momentum) {
            detail::calculateHiddenDeltas(*this, affectedLayer, momentum);
        }

        void calculateWeights(Var learningRate) {
            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                auto& neuron = (*this)[neuronId];
                const Var neuronDelta = m_deltas[neuronId];

                for(std::size_t weightId = 0; weightId < Grid::K::size; ++weightId) {
                    const Var inputValue = neuron[weightId].value;
                    const Var weightGradient = neuronDelta * inputValue;
                    auto weight = neuron.getWeight(weightId);
                    neuron.setWeight(weightId, weight - learningRate * weightGradient);
                }

                Var bias = neuron.getBias();
                neuron.setBias(bias - learningRate * neuronDelta);
            }
        }

        void accumulateGradients() {
            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                auto& neuron = (*this)[neuronId];
                const Var neuronDelta = m_deltas[neuronId];

                for(std::size_t weightId = 0; weightId < Grid::K::size; ++weightId) {
                    const Var inputValue = neuron[weightId].value;
                    m_accumulatedWeightGradients[neuronId][weightId] +=
                     neuronDelta * inputValue;
                }
                m_accumulatedBiasGradients[neuronId] += neuronDelta;
            }
        }

        void applyGradients(const Var& learningRate) {
            for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                auto& neuron = (*this)[neuronId];

                for(std::size_t weightId = 0; weightId < Grid::K::size; ++weightId) {
                    auto weight = neuron.getWeight(weightId);
                    auto gradient = m_accumulatedWeightGradients[neuronId][weightId];
                    neuron.setWeight(weightId, weight - learningRate * gradient);
                }

                Var bias = neuron.getBias();
                neuron.setBias(bias - learningRate * m_accumulatedBiasGradients[neuronId]);
            }

            resetGradients();
        }

        void resetGradients() {
            for(auto& gradients : m_accumulatedWeightGradients) {
                std::fill(gradients.begin(), gradients.end(), Var{});
            }
            std::fill(m_accumulatedBiasGradients.begin(),
                      m_accumulatedBiasGradients.end(),
                      Var{});
            std::fill(m_deltas.begin(), m_deltas.end(), Var{});
        }

        std::vector< Var >& deltas() {
            return m_deltas;
        }

        const std::vector< Var >& deltas() const {
            return m_deltas;
        }

        const Var& getDelta(std::size_t neuronId) const {
            return m_deltas[neuronId];
        }

        BPNeuralLayer()
         : m_deltas(size(), Var{}), m_accumulatedBiasGradients(size(), Var{}),
           m_accumulatedWeightGradients(size(), std::vector< Var >(Grid::K::size, Var{})) {
        }

      private:
        std::vector< Var > m_deltas;
        std::vector< Var > m_accumulatedBiasGradients;
        std::vector< std::vector< Var > > m_accumulatedWeightGradients;
    };

} // namespace nn::bp