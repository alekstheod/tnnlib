#pragma once

#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"

#include <range/v3/view.hpp>
#include <algorithm>

namespace nn::bp {
    template< typename >
    struct BPNeuralLayer;

    template< typename LayerType, typename Grid >
    struct BPNeuralLayer< nn::detail::ConvolutionLayer< LayerType, Grid > >
     : private nn::detail::ConvolutionLayer< typename LayerType::template wrap< BPNeuron >, Grid > {
        using Base =
         nn::detail::ConvolutionLayer< typename LayerType::template wrap< BPNeuron >, Grid >;

        using NeuralLayerType = typename nn::detail::ConvolutionLayer< LayerType, Grid >;

        using Var = typename NeuralLayerType::Var;

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

        void calculateWeights(Var learningRate) {
            // Process each input position in the grid
            for(const auto inputId : ranges::views::indices(Grid::width * Grid::height)) {
                const auto gradient = calculateGradient(inputId);
                adjustWeight(inputId, gradient, learningRate);
            }

            // Update biases for each neuron
            auto& self = *this;
            for(const auto i : ranges::views::indices(size())) {
                auto& neuron = self[i];
                Var currentBias = neuron.getBias();
                Var newBias = currentBias - learningRate * neuron.getDelta();
                neuron.setBias(newBias);
            }
        }

        template< typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(AffectedLayer& affectedLayer, MomentumFunc momentum) {
            using Var = typename AffectedLayer::Var;
            this->for_each([&affectedLayer, &momentum](auto i, auto& currentNeuron) {
                Var sum{}; // sum(aDelta*aWeight)
                affectedLayer.for_each([&sum, &i](auto, auto& neuron) {
                    auto affectedDelta = neuron.getDelta();
                    auto affectedWeight = neuron.getWeight(i.value);
                    sum += affectedDelta * affectedWeight;
                });

                currentNeuron.setDelta(momentum(currentNeuron.getDelta(),
                                                sum * currentNeuron.calculateDerivate()));
            });
        }

        const Var& getDelta(std::size_t neuronId) const {
            auto& self = *this;
            return self[neuronId].getDelta();
        }

      private:
        void adjustWeight(const std::size_t inputId, const Var& gradient, const Var& learningRate) {
            auto& self = *this;
            utils::for_each(m_grid.frames, [&](auto& frame) {
                if(frame.area.doesIntersect(inputId)) {
                    const auto localInputId = frame.area.localize(inputId);
                    auto& neuron = self[frame.neuronId];
                    // Use proper neuron interface methods
                    Var currentWeight = neuron.getWeight(localInputId);
                    neuron.setWeight(localInputId, currentWeight - learningRate * gradient);
                }
            });
        }

        Var calculateGradient(const std::size_t inputId) {
            Var sum{};
            auto& self = *this;

            // Accumulate gradients from all frames that use this input
            utils::for_each(m_grid.frames, [&](auto& frame) {
                if(frame.area.doesIntersect(inputId)) {
                    const auto localInputId = frame.area.localize(inputId);
                    const auto& neuron = self[frame.neuronId];
                    // Standard gradient: delta * input_value
                    sum += neuron.getDelta() * neuron[localInputId].value;
                }
            });

            return sum;
        }

        Grid m_grid;
    };

} // namespace nn::bp
