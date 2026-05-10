#pragma once

#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/Thread/AsyncNeuralLayer.h"

#include <future>

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
            using Base::calculateHiddenDeltas;
            using Base::calculateOutputs;
            using Base::for_each;
            using Base::getMemento;
            using Base::inputs;
            using Base::setMemento;
            using Base::size;
            using Base::operator[];

            template< typename BPCtx, std::size_t myIdx >
            void calculateWeights(BPCtx& ctx, const Var& learningRate) {
                auto& deltas = std::get< myIdx >(ctx.deltas);
                auto& weights = std::get< myIdx >(ctx.weights);
                auto& biases = std::get< myIdx >(ctx.biases);
                constexpr auto inputsNumber = inputs();

                std::array< std::future< void >, size() > weightFutures;

                utils::for_< size() >([&, this](const auto& i) {
                    auto delta = deltas[i.value];
                    auto idx = i.value;
                    std::promise< void > promise;
                    weightFutures[idx] = promise.get_future();
                    boost::asio::post(nn::detail::pool(),
                                      [this, idx, delta, &weights, &biases, &learningRate, promise = std::move(promise)]() mutable {
                                          for(std::size_t j = 0; j < inputsNumber; j++) {
                                              auto input = (*this)[idx][j].value;
                                              auto weight = weights[idx * inputsNumber + j];
                                              weights[idx * inputsNumber + j] = weight - learningRate * input * delta;
                                          }
                                          biases[idx] = biases[idx] - learningRate * delta;
                                          promise.set_value();
                                      });
                });

                for(auto& future : weightFutures) {
                    future.get();
                }
            }
        };

    } // namespace bp

} // namespace nn
