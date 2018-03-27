/**
*  Copyright (c) 2011, Alex Theodoridis
*  All rights reserved.

*  Redistribution and use in source and binary forms, with
*  or without modification, are permitted provided that the
*  following conditions are met:
*  Redistributions of source code must retain the above
*  copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above
*  copyright notice, this list of conditions and the following
*  disclaimer in the documentation and/or other materials
*  provided with the distribution.

*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
*  AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
*  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
*  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
*  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
*  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
*  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE,
*  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

#ifndef BEPNeuralLayerH
#define BEPNeuralLayerH

#include <NeuralNetwork/LearningAlgorithm/BackPropagation/BPNeuron.h>
#include <NeuralNetwork/INeuralLayer.h>

#include <algorithm>
#include <functional>
#include <tuple>

namespace nn {

    namespace bp {


        template< typename NeuralLayerType >
        class BPNeuralLayer
         : public nn::INeuralLayer< typename NeuralLayerType::template wrap< BPNeuron > > {
          public:
            using NeuralLayer =
             INeuralLayer< typename NeuralLayerType::template wrap< BPNeuron > >;
            using Neuron = typename NeuralLayer::Neuron;
            using Var = typename NeuralLayer::Var;

            template< typename VarType >
            using use =
             BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

            static constexpr std::size_t CONST_NEURONS_NUMBER =
             NeuralLayerType::CONST_NEURONS_NUMBER;

            template< std::size_t inputs >
            using resize =
             BPNeuralLayer< typename NeuralLayerType::template resize< inputs > >;

            /**
             * @brief Will calculate the deltas for the current layer. This
             * method must be called for the hidden layers.
             * @param affectedLayer the next affected layer.
             * @param current data set based on which the deltas will be
             * calculated.
             */
            template< typename Layer, typename MomentumFunc >
            void calculateHiddenDeltas(Layer& affectedLayer, MomentumFunc momentum) {
                std::size_t neuronId = 0;
                for(auto& neuron : *this) {
                    Var sum = 0.0f; // sum(aDelta*aWeight)
                    for(std::size_t i = 0; i < affectedLayer.size(); i++) {
                        Var affectedDelta = affectedLayer.getDelta(i);
                        Var affectedWeight = affectedLayer.getInputWeight(i, neuronId);
                        sum += affectedDelta * affectedWeight;
                        sum += affectedDelta * affectedLayer->getBias(i);
                    }

                    neuron->setDelta(momentum(neuron->getDelta(),
                                              sum * neuron->calculateDerivate()));

                    neuronId++;
                }
            }

            /**
             * @brief Will calculate the deltas for the current layer. This
             * method must be called for the output layer layers.
             * @param current data set based on which the deltas will be
             * calculated.
             */
            template< typename Prototype, typename MomentumFunc >
            void calculateDeltas(const Prototype& prototype, MomentumFunc momentum) {
                std::size_t neuronId = 0;
                for(auto curNeuron = NeuralLayer::begin();
                    curNeuron != NeuralLayer::end();
                    curNeuron++) {
                    (*curNeuron)->calculateDelta(std::get< 1 >(prototype)[neuronId], momentum);
                    neuronId++;
                }
            }

            void calculateWeights(Var learningRate) {
                for(auto& neuron : *this) {
                    std::size_t inputsNumber = neuron->size();
                    auto delta = neuron->getDelta();
                    for(std::size_t i = 0; i < inputsNumber; i++) {
                        auto input = neuron[i].value;
                        auto weight = neuron[i].weight;
                        auto newWeight = weight - learningRate * input * delta;
                        neuron.setWeight(i, newWeight);
                    }

                    Var weight = neuron->getBias();
                    Var newWeight = weight - learningRate * delta;
                    neuron->setBias(newWeight);
                }
            }

            const Var& getDelta(std::size_t neuronId) const {
                return NeuralLayer::operator[](neuronId)->getDelta();
            }
        };
    } // namespace bp
} // namespace nn

#endif
