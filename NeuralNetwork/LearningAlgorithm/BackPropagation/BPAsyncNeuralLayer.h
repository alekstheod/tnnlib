#pragma once

#include "NeuralNetwork/LearningAlgorithm/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/Thread/AsyncNeuralLayer.h"

namespace nn {

    namespace bp {

        template< typename Internal >
        struct BPNeuralLayer< nn::detail::AsyncNeuralLayer< Internal > >
         : private BPNeuralLayer< Internal > {

            using NeuralLayerType = nn::detail::AsyncNeuralLayer< Internal >;
            using Var = typename NeuralLayerType::Var;

            using Base = BPNeuralLayer< Internal >;

            template< typename VarType >
            using use =
             BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

            template< std::size_t inputs >
            using adjust =
             BPNeuralLayer< typename NeuralLayerType::template adjust< inputs > >;

            using Memento = typename Base::Memento;
            using Base::calculateDeltas;
            using Base::calculateOutputs;
            using Base::for_each;
            using Base::getMemento;
            using Base::getOutput;
            using Base::inputs;
            using Base::setInput;
            using Base::setMemento;
            using Base::size;
            using Base::operator[];

            void calculateWeights(const Var& learningRate) {
                const auto calculateWeight = [&learningRate](auto& neuron) {
                    auto delta = neuron.getDelta();
                    const std::size_t inputsNumber = neuron.size();
                    for(std::size_t i = 0; i < inputsNumber; i++) {
                        auto input = neuron[i].value;
                        auto weight = neuron[i].weight;
                        auto newWeight = weight - learningRate * input * delta;
                        neuron.setWeight(i, newWeight);
                    }

                    Var weight = neuron.getBias();
                    Var newWeight = weight - learningRate * delta;
                    neuron.setBias(newWeight);
                };


                std::array< std::future< void >, size() > weightFutures;

                utils::for_< size() >([&, this](const auto& i) {
                    auto& neuron = operator[](i.value);
                    std::promise< void > promise;
                    weightFutures[i.value] = promise.get_future();
                    boost::asio::post(nn::detail::pool(),
                                      [&neuron, promise = std::move(promise), &calculateWeight]() mutable {
                                          calculateWeight(neuron);
                                          promise.set_value();
                                      });
                });

                for(auto& future : weightFutures) {
                    future.get();
                }
            }

            const Var& getDelta(std::size_t neuronId) const {
                return Base::operator[](neuronId).getDelta();
            }
        };

    } // namespace bp

} // namespace nn
