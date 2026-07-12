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

            using Base::calculateDeltas;
            using Base::calculateHiddenDeltas;
            using Base::calculateOutputs;
            using Base::defaultConnection;
            using Base::for_each;
            using Base::inputs;
            using Base::size;
            using Base::operator[];

            template< typename BPCtx, std::size_t myIdx >
            void calculateWeights(BPCtx& ctx, const Var& learningRate) {
                auto& deltas = std::get< myIdx >(ctx.deltas);
                auto& weights = std::get< myIdx >(ctx.weights);
                auto& biases = std::get< myIdx >(ctx.biases);
                const auto& connections = std::get< myIdx >(ctx.connections);
                const auto& outputs = ctx.outputs;
                constexpr auto inputsNumber = inputs();

                std::array< std::future< void >, size() > weightFutures;

                utils::for_< size() >([&, this](const auto& i) {
                    const Var delta = deltas[i.value];
                    const auto idx = i.value;
                    std::promise< void > promise;
                    weightFutures[idx] = promise.get_future();
                    boost::asio::post(nn::detail::pool(),
                                      [this, idx, delta, &outputs, &connections, &weights, &biases, &learningRate, promise = std::move(promise)]() mutable {
                                          const auto& neuronConn = connections[idx];
                                          for(std::size_t s = 0; s < inputsNumber; ++s) {
                                              if constexpr (myIdx > 0) {
                                                  if(const Var* input = outputOf< Var >(outputs, neuronConn[s])) {
                                                      weights[idx * inputsNumber + s] -=
                                                       learningRate * (*input) * delta;
                                                  }
                                              } else {
                                                  const Var input = (*this)[idx][s].value;
                                                  weights[idx * inputsNumber + s] -=
                                                   learningRate * input * delta;
                                              }
                                          }
                                          biases[idx] -= learningRate * delta;
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
