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
            auto& self = *this;
            for(const auto neuronId : ranges::views::indices(size())) {
                auto& neuron = self[neuronId];
                const Var neuronDelta = neuron.getDelta();

                for(const auto weightId : ranges::views::indices(Grid::K::size)) {
                    const Var inputValue = neuron[weightId].value;
                    const Var weightGradient = neuronDelta * inputValue;
                    neuron[weightId].weight =
                     neuron[weightId].weight - learningRate * weightGradient;
                }

                Var bias = neuron.getBias();
                Var newBias = bias - learningRate * neuronDelta;
                neuron.setBias(newBias);
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
        Grid m_grid;
    };

} // namespace nn::bp
