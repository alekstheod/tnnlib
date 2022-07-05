#pragma once


#include <NeuralNetwork/LearningAlgorithm/BackPropagation/BPNeuralLayer.h>
#include <NeuralNetwork/NeuralLayer/ConvolutionLayer.h>
#include <NeuralNetwork/INeuralLayer.h>

#include <algorithm>

namespace nn {

    namespace bp {

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
                    auto delta = neuron->getDelta();

                    auto calculateWeight =
                     [&](const std::array< Var, Grid::frameSize >& weightIdxs) {
                         for(auto inputId : ranges::views::indices(neuron->size())) {
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

                    Var weight = neuron->getBias();
                    Var newWeight = weight - learningRate * delta;
                    neuron->setBias(newWeight);
                }

                for(auto neuronId : ranges::views::indices(this->size())) {
                    auto& neuron = (*this)[neuronId];
                    for(auto inputId : ranges::views::indices(neuron->size())) {
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
