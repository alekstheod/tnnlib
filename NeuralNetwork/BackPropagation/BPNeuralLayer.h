#pragma once

#include "NeuralNetwork/BackPropagation/ErrorFunction.h"

#include <MPL/TypeTraits.h>

#include <range/v3/all.hpp>
#include <vector>


namespace nn::bp {
    template< typename Internal >
    struct BPNeuralLayer;

    namespace detail {

        template< typename CurrentLayer, typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(CurrentLayer& currentLayer,
                                   AffectedLayer& affectedLayer,
                                   MomentumFunc momentum) {
            using Var = typename AffectedLayer::Var;
            auto& currentDeltas = currentLayer.deltas();
            auto& affectedDeltas = affectedLayer.deltas();

            currentLayer.for_each([&affectedLayer,
                                   &currentLayer,
                                   &currentDeltas,
                                   &affectedDeltas,
                                   &momentum](auto i, auto& currentNeuron) {
                Var sum{};
                affectedLayer.for_each([&sum, &i, &affectedDeltas](auto j, auto& neuron) {
                    auto affectedDelta = affectedDeltas[j.value];
                    auto affectedWeight = neuron.getWeight(i.value);
                    sum += affectedDelta * affectedWeight;
                });

                auto currentDerivative =
                 currentNeuron.getOutputFunction().derivate(currentNeuron.getOutput());
                currentDeltas[i.value] =
                 momentum(currentDeltas[i.value], sum * currentDerivative);
            });
        }

    } // namespace detail


    template< typename NeuralLayerType >
    struct BPNeuralLayer : NeuralLayerType {
        using Base = NeuralLayerType;

        using NeuralLayer = NeuralLayerType;
        using Var = typename NeuralLayer::Var;

        template< typename VarType >
        using use = BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

        template< std::size_t inputs >
        using adjust =
         BPNeuralLayer< typename NeuralLayerType::template adjust< inputs > >;

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
        using Base::operator[];
        using Base::calculateOutputs;
        using Base::getMemento;
        using Base::getOutput;
        using Base::setInput;
        using Base::setMemento;


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

        void calculateWeights(const Var& learningRate) {
            for_each([this, learningRate](auto i, auto& neuron) {
                auto delta = m_deltas[i.value];
                for(std::size_t j = 0; j < inputs(); ++j) {
                    auto input = neuron[j].value;
                    auto weight = neuron[j].weight;
                    neuron[j].weight = weight - learningRate * input * delta;
                }

                auto bias = neuron.getBias();
                neuron.setBias(bias - learningRate * delta);
            });
        }

        void accumulateGradients() {
            for_each([this](auto i, auto& neuron) {
                auto delta = m_deltas[i.value];
                for(std::size_t j = 0; j < inputs(); ++j) {
                    auto input = neuron[j].value;
                    m_accumulatedWeightGradients[i.value][j] += input * delta;
                }
                m_accumulatedBiasGradients[i.value] += delta;
            });
        }

        void applyGradients(const Var& learningRate) {
            for_each([this, learningRate](auto i, auto& neuron) {
                for(std::size_t j = 0; j < inputs(); ++j) {
                    auto weight = neuron[j].weight;
                    auto gradient = m_accumulatedWeightGradients[i.value][j];
                    neuron[j].weight = weight - learningRate * gradient;
                }

                auto bias = neuron.getBias();
                neuron.setBias(bias - learningRate * m_accumulatedBiasGradients[i.value]);
            });

            resetGradients();
        }

        void resetGradients() {
            for(auto& neuronGradients : m_accumulatedWeightGradients) {
                std::fill(neuronGradients.begin(), neuronGradients.end(), Var{});
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
           m_accumulatedWeightGradients(size(), std::vector< Var >(inputs(), Var{})) {
        }

      private:
        std::vector< Var > m_deltas;
        std::vector< Var > m_accumulatedBiasGradients;
        std::vector< std::vector< Var > > m_accumulatedWeightGradients;
    };
} // namespace nn::bp