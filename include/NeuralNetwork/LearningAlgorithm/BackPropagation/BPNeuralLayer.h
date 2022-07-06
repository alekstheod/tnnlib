#pragma once

#include <NeuralNetwork/LearningAlgorithm/BackPropagation/BPNeuron.h>
#include <NeuralNetwork/NeuralLayer/ConvolutionLayer.h>

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
         : public detail::unwrapLayer< typename NeuralLayerType::template wrap< BPNeuron > >::type {
          public:
            using Base =
             typename detail::unwrapLayer< typename NeuralLayerType::template wrap< BPNeuron > >::type;
            using NeuralLayer = typename detail::unwrapLayer< NeuralLayerType >::type;
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
    } // namespace bp
} // namespace nn
