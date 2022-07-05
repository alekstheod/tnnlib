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
#include <NeuralNetwork/NeuralLayer/ConvolutionLayer.h>
#include <NeuralNetwork/INeuralLayer.h>

#include <MPL/TypeTraits.h>

#include <range/v3/all.hpp>

#include <algorithm>
#include <functional>
#include <tuple>
#include <type_traits>

namespace nn {

    namespace bp {
        template< typename Internal >
        class BPNeuralLayer;

        namespace detail {
            template< typename Internal >
            struct unwrapLayer {
                using type = Internal;
            };

            template< typename Internal >
            struct unwrapLayer< BPNeuralLayer< Internal > > {
                using type = typename unwrapLayer< Internal >::type;
            };

            template< typename It, typename Layer, typename MomentumFunc >
            void calculateHiddenDeltas(It begin, It end, Layer& affectedLayer, MomentumFunc momentum) {
                auto current = begin;
                using Var = typename Layer::Var;
                while(current != end) {
                    Var sum{}; // sum(aDelta*aWeight)
                    for(const auto& neuron : affectedLayer) {
                        auto affectedDelta = neuron.getDelta();
                        auto affectedWeight = neuron[current - begin].weight;
                        sum += affectedDelta * affectedWeight;
                        sum += affectedDelta * neuron.getBias();
                    }

                    current->setDelta(momentum(current->getDelta(),
                                               sum * current->calculateDerivate()));

                    current = std::next(current);
                }
            }
        } // namespace detail

        template< typename NeuralLayerType >
        class BPNeuralLayer
         : public nn::INeuralLayer< typename detail::unwrapLayer< typename NeuralLayerType::template wrap< BPNeuron > >::type > {
          public:
            using Base =
             INeuralLayer< typename detail::unwrapLayer< typename NeuralLayerType::template wrap< BPNeuron > >::type >;
            using NeuralLayer =
             INeuralLayer< typename detail::unwrapLayer< NeuralLayerType >::type >;
            using Var = typename NeuralLayer::Var;

            template< typename VarType >
            using use =
             BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

            static constexpr std::size_t CONST_NEURONS_NUMBER =
             NeuralLayerType::CONST_NEURONS_NUMBER;

            template< std::size_t inputs >
            using adjust =
             BPNeuralLayer< typename NeuralLayerType::template adjust< inputs > >;

            /**
             * @brief Will calculate the deltas for the current layer. This
             * method must be called for the hidden layers.
             * @param affectedLayer the next affected layer.
             * @param current data set based on which the deltas will be
             * calculated.
             */
            template< typename Layer, typename MomentumFunc >
            void calculateHiddenDeltas(Layer& affectedLayer, MomentumFunc momentum) {
                detail::calculateHiddenDeltas(this->begin(), this->end(), affectedLayer, momentum);
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
                for(auto& neuron : *this) {
                    neuron.calculateDelta(std::get< 1 >(prototype)[neuronId], momentum);
                    neuronId++;
                }
            }

            void calculateWeights(Var learningRate) {
                for(auto& neuron : *this) {
                    std::size_t inputsNumber = neuron.size();
                    auto delta = neuron.getDelta();
                    for(std::size_t i = 0; i < inputsNumber; i++) {
                        auto input = neuron[i].value;
                        auto weight = neuron[i].weight;
                        auto newWeight = weight - learningRate * input * delta;
                        neuron.setWeight(i, newWeight);
                    }

                    Var weight = neuron.getBias();
                    Var newWeight = weight - learningRate * delta;
                    neuron.setBias(newWeight);
                }
            }

            const Var& getDelta(std::size_t neuronId) const {
                return Base::operator[](neuronId)->getDelta();
            }
        };

        template< typename LayerType, typename Grid >
        class BPNeuralLayer< nn::detail::ConvolutionLayer< LayerType, Grid > >
         : public nn::INeuralLayer< nn::detail::ConvolutionLayer< typename LayerType::template wrap< BPNeuron >, Grid > > {
          public:
            using Base =
             nn::INeuralLayer< nn::detail::ConvolutionLayer< typename LayerType::template wrap< BPNeuron >, Grid > >;

            using NeuralLayerType =
             typename nn::detail::ConvolutionLayer< LayerType, Grid >;

            using Var = typename NeuralLayerType::Var;

            template< typename VarType >
            using use =
             BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

            static constexpr std::size_t CONST_NEURONS_NUMBER =
             NeuralLayerType::CONST_NEURONS_NUMBER;

            template< std::size_t inputs >
            using adjust = BPNeuralLayer;

            using Base::size;

            BPNeuralLayer() {
                auto firstNeuron = this->begin();
                for(std::size_t i = 0u; i < Grid::frameSize; i++) {
                    m_weights[i] = firstNeuron->getWeight(i);
                    m_originalIdx[i] = i;
                }

                m_reversedIdx = m_originalIdx;
                m_flippedIdx = m_originalIdx;

                std::reverse(m_reversedIdx.begin(), m_reversedIdx.end());
                m_reversedFlippedIdx = m_reversedIdx;
                for(auto i = 0u; i < Grid::frameSize; i += Grid::filterWidth) {
                    std::reverse(m_flippedIdx.begin() + i,
                                 m_flippedIdx.begin() + i + Grid::filterWidth);

                    std::reverse(m_reversedFlippedIdx.begin() + i,
                                 m_reversedFlippedIdx.begin() + i + Grid::filterWidth);
                }

                for(auto& neuron : *this) {
                    for(auto i = 0u; i < Grid::frameSize; i++) {
                        neuron.setWeight(i, m_weights[i]);
                    }
                }
            }
            /**
             * @brief Will calculate the deltas for the current layer. This
             * method must be called for the hidden layers.
             * @param affectedLayer the next affected layer.
             * @param current data set based on which the deltas will be
             * calculated.
             */
            template< typename Layer, typename MomentumFunc >
            void calculateHiddenDeltas(Layer& affectedLayer, MomentumFunc momentum) {
                detail::calculateHiddenDeltas(this->begin(), this->end(), affectedLayer, momentum);
            }

            void calculateWeights(Var learningRate) {
                for(auto neuronId : ranges::views::indices(this->size())) {
                    auto& neuron = (*this)[neuronId];
                    auto delta = neuron.getDelta();

                    auto calculateWeight =
                     [&](const std::array< Var, Grid::frameSize >& weightIdxs) {
                         for(auto inputId : ranges::views::indices(neuron.size())) {
                             auto input = neuron[inputId].value;
                             auto weight = neuron[inputId].weight;
                             m_weights[weightIdxs[inputId]] +=
                              (weight - learningRate * input * delta);
                         }
                     };

                    if((neuronId / Grid::rowSize) % 2) {
                        if(neuronId % Grid::rowSize % 2) {
                            calculateWeight(m_reversedFlippedIdx);
                        } else {
                            calculateWeight(m_flippedIdx);
                        }
                    } else {
                        if(neuronId % Grid::rowSize % 2) {
                            calculateWeight(m_reversedIdx);
                        } else {
                            calculateWeight(m_originalIdx);
                        }
                    }

                    Var weight = neuron.getBias();
                    Var newWeight = weight - learningRate * delta;
                    neuron.setBias(newWeight);
                }

                for(auto neuronId : ranges::views::indices(size())) {
                    auto& neuron = (*this)[neuronId];
                    for(auto inputId : ranges::views::indices(neuron.size())) {
                        neuron.setWeight(inputId, m_weights[inputId]);
                    }
                }
            }

            const Var& getDelta(std::size_t neuronId) const {
                return Base::operator[](neuronId)->getDelta();
            }

          private:
            std::array< Var, Grid::frameSize > m_originalIdx;
            std::array< Var, Grid::frameSize > m_reversedIdx;
            std::array< Var, Grid::frameSize > m_flippedIdx;
            std::array< Var, Grid::frameSize > m_reversedFlippedIdx;

            std::array< Var, Grid::frameSize > m_weights;
        }; // namespace bp
    } // namespace bp
} // namespace nn

#endif
