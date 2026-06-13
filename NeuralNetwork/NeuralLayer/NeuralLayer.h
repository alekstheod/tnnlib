#pragma once

#include "NeuralNetwork/NeuralLayer/Container.h"
#include "NeuralNetwork/NeuralLayer/Vector.h"
#include "NeuralNetwork/NeuralLayer/Tuple.h"

#include <MPL/Algorithm.h>

#include <array>

namespace nn {

    namespace detail {
        /**
         * Represent the NeuralLayer in perceptron.
         */
        template< typename T >
        struct NeuralLayer : public Layer< T > {
            using Base = Layer< T >;

          public:
            using Container = typename Base::Container;
            using Var = typename Base::Var;
            using ActivationFunctions = typename Base::ActivationFunctions;

            template< template< class > typename NewType >
            using wrap = typename Base::template wrap_neuron< NeuralLayer, NewType >;

            template< unsigned int inputs >
            using adjust = typename Base::template adjust_inputs< NeuralLayer, inputs >;

            template< typename VarType >
            using use = typename Base::template use_var< NeuralLayer, VarType >;

            template< typename NewNeuron >
            using with_neuron = typename Base::template with_neuron< NewNeuron >;

            using Input = Var;

            using Base::begin;
            using Base::cbegin;
            using Base::cend;
            using Base::end;
            using Base::inputs;
            using Base::size;
            using Base::operator[];

            static_assert(size() > 0,
                          "Invalid template argument neuronsNumber == 0");
            static_assert(inputs() >= 1,
                          "Invalid template argument inputsNumber <= 1");

            template< typename Func >
            void for_each(Func func) {
                utils::for_< size() >([this, &func](auto i) {
                    func(i, utils::get< i.value >(m_neurons));
                });
            }

            template< typename Func >
            void for_each(Func func) const {
                utils::for_< size() >([this, &func](auto i) {
                    func(i, utils::get< i.value >(m_neurons));
                });
            }

            template< typename Context, std::size_t myIdx, std::size_t predecessorIdx >
            void calculateOutputs(Context& ctx) {
                auto& predecessorOutputs = std::get< predecessorIdx >(ctx);
                auto& myOutputs = std::get< myIdx >(ctx);
                constexpr auto inputCount = inputs();
                const auto predSize = predecessorOutputs.size();

                std::array< Var, size() > dotProducts;
                utils::for_< size() >([this, &predecessorOutputs, &dotProducts, predSize](auto i) {
                    auto& neuron = utils::get< i.value >(m_neurons);
                    dotProducts[i.value] =
                     neuron.calcDotProduct(predecessorOutputs.data(),
                                           predSize < inputCount ? predSize : inputCount);
                });

                utils::for_< size() >([this, &dotProducts, &myOutputs](auto i) {
                    const auto output = utils::get< i.value >(m_neurons).calculateOutput(
                     dotProducts[i.value], std::cbegin(dotProducts), std::cend(dotProducts));
                    myOutputs[i.value] = output;
                });
            }

            template< typename Context, std::size_t myIdx, std::size_t predecessorIdx, typename W >
            void calculateOutputs(Context& ctx, const W& wctx) {
                auto& predecessorOutputs = std::get< predecessorIdx >(ctx);
                auto& myOutputs = std::get< myIdx >(ctx);
                const auto& weights = std::get< myIdx >(wctx.weights);
                const auto& biases = std::get< myIdx >(wctx.biases);
                constexpr auto neuronInputs = inputs();
                const auto inputSize = predecessorOutputs.size() < neuronInputs
                                       ? predecessorOutputs.size() : neuronInputs;

                std::array< Var, size() > dotProducts;
                utils::for_< size() >([&](auto i) {
                    Var dot = biases[i.value];
                    for (std::size_t j = 0; j < inputSize; ++j) {
                        dot += predecessorOutputs[j] * weights[i.value * neuronInputs + j];
                    }
                    dotProducts[i.value] = dot;
                });

                utils::for_< size() >([this, &dotProducts, &myOutputs](auto i) {
                    const auto output = utils::get< i.value >(m_neurons).calculateOutput(
                     dotProducts[i.value], std::cbegin(dotProducts), std::cend(dotProducts));
                    myOutputs[i.value] = output;
                });
            }

            template< typename Context, std::size_t myIdx >
            void calculateOutputs(Context& ctx) {
                auto& myOutputs = std::get< myIdx >(ctx);

                std::array< Var, size() > dotProducts;
                utils::for_< size() >([this, &dotProducts](auto i) {
                    dotProducts[i.value] =
                     utils::get< i.value >(m_neurons).calcDotProduct();
                });

                utils::for_< size() >([this, &dotProducts, &myOutputs](auto i) {
                    const auto output = utils::get< i.value >(m_neurons).calculateOutput(
                     dotProducts[i.value], std::cbegin(dotProducts), std::cend(dotProducts));
                    myOutputs[i.value] = output;
                });
            }

          private:
            using Base::m_neurons;
            NeuralLayer& self{*this};
        };
    } // namespace detail

    template< template< template< class > class, class, std::size_t > class NeuronType,
              template< class > class ActivationFunctionType,
              std::size_t size,
              std::size_t inputsNumber = size,
              typename Var = float >
    using NeuralLayer =
     detail::NeuralLayer< nn::detail::Vector< NeuronType< ActivationFunctionType, Var, inputsNumber >, size > >;

    template< std::size_t inputs, typename Var, typename... Neuron >
    using ComplexNeuralInputLayer =
     detail::NeuralLayer< detail::Tuple< Var, inputs, typename Neuron::template resize< inputs >... > >;

    template< typename... Neuron >
    using ComplexNeuralLayer =
     detail::NeuralLayer< detail::Tuple< float, 1, Neuron... > >;
} // namespace nn
