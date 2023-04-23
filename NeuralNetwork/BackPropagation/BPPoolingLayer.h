#pragma once

#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/PoolingLayer.h"

#include <range/v3/view.hpp>
#include <algorithm>

namespace nn::bp {
    template< typename >
    struct BPNeuralLayer;

    template< typename Internal >
    struct BPNeuralLayer< nn::detail::PoolingLayer< Internal > >
     : private nn::detail::PoolingLayer< typename Internal::template wrap< BPNeuron > > {
        using Base =
         nn::detail::PoolingLayer< typename Internal::template wrap< BPNeuron > >;

        using Var = typename Internal::Var;

        template< typename VarType >
        using use = BPNeuralLayer< typename Base::template use< VarType > >;

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
        }

        template< typename AffectedLayer, typename MomentumFunc >
        void calculateHiddenDeltas(AffectedLayer& affectedLayer, MomentumFunc momentum) {
            // detail::calculateHiddenDeltas(*this, affectedLayer, momentum);
        }

        const Var& getDelta(std::size_t neuronId) const {
            auto& self = *this;
            return self[neuronId].getDelta();
        }
    };

} // namespace nn::bp
