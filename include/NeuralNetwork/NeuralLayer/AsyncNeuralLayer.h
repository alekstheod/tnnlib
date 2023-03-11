#pragma once

#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>

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
                std::array< std::future< Var >, size() > dotProducts;
                utils::for_< size() >([this, &dotProducts](auto i) {
                    auto& neuron = operator[](i.value);
                    dotProducts[i.value] =
                     std::async([&neuron]() { return neuron.calcDotProduct(); });
                });

                const auto products =
                 dotProducts |
                 ranges::views::transform(std::mem_fn(&std::future< Var >::get));
                for_each([&products](auto, auto& neuron) {
                    neuron.calculateOutput(std::cbegin(products), std::cend(products));
                });
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
