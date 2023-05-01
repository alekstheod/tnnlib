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
            for(const auto inputId : ranges::views::indices(Grid::width * Grid::height)) {
                const auto gradient = calculateGradient(inputId);
                adjustWeight(inputId, gradient, learningRate);
            }

            auto& self = *this;
            for(const auto i : ranges::views::indices(size())) {
                auto& neuron = self[i];
                Var weight = neuron.getBias();
                Var newWeight = weight - learningRate * neuron.getDelta();
                neuron.setBias(newWeight);
            }
        }

        template< typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(AffectedLayer& affectedLayer, MomentumFunc momentum) {
            detail::calculateHiddenDeltas(*this, affectedLayer, momentum);
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
                    neuron[localInputId].weight =
                     neuron[localInputId].weight - learningRate * gradient;
                }
            });
        }

        Var calculateGradient(const std::size_t inputId) {
            Var sum{};
            auto& self = *this;
            utils::for_each(m_grid.frames, [&](auto& frame) {
                if(frame.area.doesIntersect(inputId)) {
                    const auto localInputId = frame.area.localize(inputId);
                    const auto& neuron = self[frame.neuronId];
                    sum += neuron.getDelta() * neuron[localInputId].value;
                }
            });

            return sum;
        }

        Grid m_grid;
    };

} // namespace nn::bp
