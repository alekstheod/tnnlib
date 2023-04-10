#pragma once

#include "NeuralNetwork/LearningAlgorithm/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"

#include <algorithm>

namespace nn {

    namespace bp {

        template< typename LayerType, typename Grid >
        struct BPNeuralLayer< nn::detail::ConvolutionLayer< LayerType, Grid > >
         : nn::detail::ConvolutionLayer< typename LayerType::template wrap< BPNeuron >, Grid > {
            using Base =
             nn::detail::ConvolutionLayer< typename LayerType::template wrap< BPNeuron >, Grid >;

            using NeuralLayerType =
             typename nn::detail::ConvolutionLayer< LayerType, Grid >;

            using Var = typename NeuralLayerType::Var;

            template< typename VarType >
            using use =
             BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

            template< std::size_t inputs >
            using adjust = BPNeuralLayer;

            using Base::for_each;
            using Base::inputs;
            using Base::size;

            void calculateWeights(Var learningRate) {
                for(const auto inputId : ranges::views::indices(Grid::width * Grid::height)) {
                    const auto gradient = calculateGradient(inputId);
                    adjustWeight(inputId, gradient, learningRate);
                }

                auto& self = *this;
                for(const auto i : ranges::views::indices(size())) {
                    auto& neuron = self[i];
                    Var weight = neuron.getBias();
                    Var newWeight = weight - learningRate * neuron.getDelta();
                    neuron.setBias(newWeight);
                }
            }

            const Var& getDelta(std::size_t neuronId) const {
                auto& self = *this;
                return self[neuronId].getDelta();
            }

          private:
            void adjustWeight(const std::size_t inputId, const Var& gradient, const Var& learningRate) {
                auto& self = *this;
                utils::for_each(m_grid.connections, [&](auto& connection) {
                    if(connection.area.doesIntersect(inputId)) {
                        const auto localInputId = connection.area.localize(inputId);
                        auto& neuron = self[connection.neuronId];
                        neuron[localInputId].weight =
                         neuron[localInputId].weight - learningRate * gradient;
                    }
                });
            }

            Var calculateGradient(const std::size_t inputId) {
                Var sum{};
                auto& self = *this;
                utils::for_each(m_grid.connections, [&](auto& connection) {
                    if(connection.area.doesIntersect(inputId)) {
                        const auto localInputId = connection.area.localize(inputId);
                        const auto& neuron = self[connection.neuronId];
                        sum += neuron.getDelta() * neuron[localInputId].value;
                    }
                });

                return sum;
            }

            Grid m_grid;
            std::array< Var, inputs() > m_weights;
        }; // namespace bp
    } // namespace bp
} // namespace nn
