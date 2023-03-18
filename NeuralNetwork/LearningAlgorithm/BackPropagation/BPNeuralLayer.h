#pragma once

#include "NeuralNetwork/LearningAlgorithm/BackPropagation/BPNeuron.h"
#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"

#include <MPL/TypeTraits.h>

#include <range/v3/all.hpp>

#include <algorithm>
#include <functional>
#include <tuple>
#include <type_traits>

namespace nn {

    namespace bp {
        template< typename Internal >
        struct BPNeuralLayer;

        namespace detail {
            template< typename Internal >
            struct unwrapLayer {
                using type = Internal;
            };

            template< typename Internal >
            struct unwrapLayer< BPNeuralLayer< Internal > > {
                using type = typename unwrapLayer< Internal >::type;
            };

            template< typename CurrentLayer, typename AffectedLayer, typename MomentumFunc >
            void calculateHiddenDeltas(CurrentLayer& currentLayer,
                                       AffectedLayer& affectedLayer,
                                       MomentumFunc momentum) {
                using Var = typename AffectedLayer::Var;
                currentLayer.for_each([&affectedLayer, &momentum](auto i, auto& currentNeuron) {
                    Var sum{}; // sum(aDelta*aWeight)
                    affectedLayer.for_each([&sum, &i](auto, auto& neuron) {
                        auto affectedDelta = neuron.getDelta();
                        auto affectedWeight = neuron[i.value].weight;
                        sum += affectedDelta * affectedWeight;
                        sum += affectedDelta * neuron.getBias();
                    });

                    currentNeuron.setDelta(momentum(currentNeuron.getDelta(),
                                                    sum * currentNeuron.calculateDerivate()));
                });
            }
        } // namespace detail

        template< typename NeuralLayerType >
        struct BPNeuralLayer
         : detail::unwrapLayer< typename NeuralLayerType::template wrap< BPNeuron > >::type {
            using Base =
             typename detail::unwrapLayer< typename NeuralLayerType::template wrap< BPNeuron > >::type;

            using NeuralLayer = typename detail::unwrapLayer< NeuralLayerType >::type;
            using Var = typename NeuralLayer::Var;

            template< typename VarType >
            using use =
             BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

            template< std::size_t inputs >
            using adjust =
             BPNeuralLayer< typename NeuralLayerType::template adjust< inputs > >;

            using Base::for_each;

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

            void calculateWeights(const Var& learningRate) {
                for_each([&learningRate](auto, auto& neuron) {
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
                });
            }

            const Var& getDelta(std::size_t neuronId) const {
                return Base::operator[](neuronId).getDelta();
            }
        };
    } // namespace bp
} // namespace nn
