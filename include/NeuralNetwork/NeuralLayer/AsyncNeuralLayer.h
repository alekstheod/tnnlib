#pragma once

#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>

#include <range/v3/all.hpp>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include <range/v3/all.hpp>

#include <thread>
#include <future>

namespace nn {

    namespace detail {
        template< typename Internal >
        struct AsyncNeuralLayer : private Internal {
            using Internal::begin;
            using Internal::cbegin;
            using Internal::cend;
            using Internal::end;
            using Internal::for_each;
            using Internal::getMemento;
            using Internal::getOutput;
            using Internal::inputs;
            using Memento = typename Internal::Memento;
            using Var = typename Internal::Var;
            using Internal::operator[];
            using Internal::setInput;
            using Internal::setMemento;
            using Internal::size;

            template< template< class > class NewType >
            using wrap =
             AsyncNeuralLayer< typename Internal::template wrap< NewType > >;

            template< unsigned int inputs >
            using adjust =
             AsyncNeuralLayer< typename Internal::template adjust< inputs > >;

            template< typename VarType >
            using use = AsyncNeuralLayer< typename Internal::template use< VarType > >;

            template< typename Layer >
            void calculateOutputs(Layer& nextLayer) {
                calculateOutputs();
                for(unsigned int i = 0; i < size(); i++) {
                    nextLayer.setInput(i, operator[](i).getOutput());
                }
            }

            void calculateOutputs() {
                std::array< std::future< Var >, size() > dotFuture;

                utils::for_< size() >([this, &dotFuture](auto i) {
                    std::promise< Var > promise;
                    dotFuture[i.value] = promise.get_future();
                    auto& neuron = operator[](i.value);
                    boost::asio::post(pool(), [&neuron, promise = std::move(promise)]() mutable {
                        promise.set_value(neuron.calcDotProduct());
                    });
                });

                auto products = dotFuture | ranges::views::transform([](auto& future) {
                                    return future.get();
                                });

                for_each([&products](auto, auto& neuron) {
                    neuron.calculateOutput(std::cbegin(products), std::cend(products));
                });
            }

          private:
            boost::asio::thread_pool& pool(std::size_t numberOfThreads = 8) {
                static boost::asio::thread_pool threadPool{numberOfThreads};
                return threadPool;
            }
        };
    } // namespace detail

    /// @brief neural layer (utilizes thread pool to compute a dot product)
    /// @param NeuronType a type of the neuron in a layer.
    /// @param ActivationFunction a type of the activation function used in a
    /// neuron.
    /// @param size ammount of neurons in a layer.
    /// @param inputsNumber the number of inputs of each neuron in a layer.
    /// initialization a final weight will be calculated in a following way
    /// random(0, 1)/scaleFactor
    template< template< template< class > class, class, std::size_t > class NeuronType,
              template< class >
              class ActivationFunctionType,
              std::size_t size,
              std::size_t inputsNumber = 2,
              typename Var = float >
    using AsyncNeuralLayer =
     detail::AsyncNeuralLayer< NeuralLayer< NeuronType, ActivationFunctionType, size, inputsNumber > >;
} // namespace nn
