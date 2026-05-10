#pragma once

#include "NeuralNetwork/BackPropagation/BPContext.h"
#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/BPConvolutionNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/ErrorFunction.h"
#include <System/Time.h>

#include <algorithm>
#include <array>
#include <chrono>

namespace nn::bp {

    template< typename PerceptronType, template< class > class ErrorCalculator = SquaredError >
    class BepAlgorithm {
        using Var = typename PerceptronType::VarType;
        using Input = typename PerceptronType::Input;

        static constexpr unsigned int inputsNumber = PerceptronType::inputs();
        static constexpr unsigned int outputsNumber = PerceptronType::outputs();

        using Perceptron = typename PerceptronType::template wrap< BPNeuralLayer >;
        using Layers = typename Perceptron::Layers;
        using BPCtx = BPContext< Var, Layers >;

      public:
        using Prototype =
         typename std::tuple< std::array< Input, inputsNumber >, std::array< Var, outputsNumber > >;
        using Memento = typename Perceptron::Memento;


        static constexpr auto size() {
            return PerceptronType::size();
        }

        /// @brief constructor will initialize the object with a learning
        /// rate and maximum error limit.
        /// @param varP the learning rate.
        /// @param maxError the limit for the error. Algorithm will stop
        /// when we reach the limit.
        BepAlgorithm(Var learningRate)
         : m_leariningRate(learningRate)
         , m_bpContext{m_perceptron.context(), {}, {}, {}, {}, {}} {
            copyWeightsToContext();
        }

        /// @brief execution of the single learning step in this algorithm.
        /// @param prototype a prototype used for this step.
        /// @param momentum a callback which will calculate a new delta,
        /// used in order to introduce momentum.
        /// @return error on this step.
        template< typename MomentumFunc >
        Var executeTrainingStep(const Prototype& prototype, MomentumFunc momentum) {
            forwardPass(std::get< 0 >(prototype).begin(),
                        std::get< 0 >(prototype).end(),
                        m_outputs.begin());

            calculateDelta(m_bpContext, prototype, momentum);

            utils::for_< size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(m_perceptron.layers());
                hiddenLayer.template calculateWeights< BPCtx, i.value + 1 >(m_bpContext, m_leariningRate);
            });

            return m_errorCalculator(m_outputs.begin(),
                                     m_outputs.end(),
                                     std::get< 1 >(prototype).begin());
        }

        template< typename MomentumFunc >
        Var executeBatchTrainingStep(const Prototype& prototype, MomentumFunc momentum) {
            forwardPass(std::get< 0 >(prototype).begin(),
                        std::get< 0 >(prototype).end(),
                        m_outputs.begin());

            calculateDelta(m_bpContext, prototype, momentum);

            utils::for_< size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(m_perceptron.layers());
                hiddenLayer.template accumulateGradients< BPCtx, i.value + 1 >(m_bpContext);
            });

            return m_errorCalculator(m_outputs.begin(),
                                     m_outputs.end(),
                                     std::get< 1 >(prototype).begin());
        }

        void applyBatchGradients() {
            utils::for_< size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(m_perceptron.layers());
                hiddenLayer.template applyGradients< BPCtx, i.value + 1 >(m_bpContext, m_leariningRate);
            });
        }

        template< typename Iterator, typename BatchErrorFunc >
        PerceptronType calculateWithBatchTraining(Iterator begin,
                                                  Iterator end,
                                                  std::size_t batchSize,
                                                  BatchErrorFunc batchErrorFunc) {
            return calculateWithBatchTraining(begin, end, batchSize, batchErrorFunc, DummyMomentum());
        }

        template< typename Iterator, typename BatchErrorFunc, typename MomentumFunc >
        PerceptronType calculateWithBatchTraining(Iterator begin,
                                                  Iterator end,
                                                  std::size_t batchSize,
                                                  BatchErrorFunc batchErrorFunc,
                                                  MomentumFunc momentum) {
            unsigned int epochCounter = 0;
            typename std::vector< Prototype > prototypes(begin, end);

            Var error{};
            do {
                auto seed = std::chrono::system_clock::now().time_since_epoch().count();

                std::vector< int > idxs(prototypes.size());
                std::iota(std::begin(idxs), std::end(idxs), 0);
                std::shuffle(std::begin(idxs),
                             std::end(idxs),
                             std::default_random_engine(static_cast< unsigned int >(seed)));

                error = {};
                for(std::size_t i = 0; i < prototypes.size(); i += batchSize) {
                    error = {};
                    for(std::size_t j = i;
                        j < std::min(i + batchSize, prototypes.size());
                        ++j) {
                        error += executeBatchTrainingStep(prototypes[j], momentum);
                    }
                    applyBatchGradients();
                }

            } while(batchErrorFunc(++epochCounter, error / prototypes.size()));

            copyWeightsFromContextToNeurons();
            PerceptronType result;
            utils::for_< PerceptronType::size() - 1 >([this, &result](auto i) {
                auto& srcLayer = std::get< i.value + 1 >(m_perceptron.layers());
                auto& dstLayer = utils::get< i.value + 1 >(result.layers());
                dstLayer.setMemento(srcLayer.getMemento());
            });

            return result;
        }

        template< typename Iterator, typename ErrorFunc >
        PerceptronType calculate(Iterator begin, Iterator end, ErrorFunc func) {
            return calculate(begin, end, func, DummyMomentum());
        }

        void setMemento(Memento memento) {
            m_perceptron.setMemento(memento);
            copyWeightsToContext();
        }

        /// @brief will calculate a perceptron with appropriate weights.
        /// @param begin iterator which points to the first input.
        /// @param end iterator which points to the last input.
        /// @param ReportFunc error report function (callback).
        /// @param MomentumFunc function which will calculate a momentum.
        /// @return a calculated perceptron.
        template< typename Iterator, typename ErrorFunc, typename MomentumFunc >
        PerceptronType calculate(Iterator begin,
                                 Iterator end,
                                 ErrorFunc errorFunc,
                                 MomentumFunc momentum = DummyMomentum()) {
            unsigned int epochCounter = 0;
            typename std::vector< Prototype > prototypes(begin, end);

            Var error{};
            do {
                auto seed = std::chrono::system_clock::now().time_since_epoch().count();

                std::vector< int > idxs(prototypes.size());
                std::iota(std::begin(idxs), std::end(idxs), 0);
                std::shuffle(std::begin(idxs),
                             std::end(idxs),
                             std::default_random_engine(static_cast< unsigned int >(seed)));

                error = {};
                for(auto idx : idxs) {
                    error += executeTrainingStep(prototypes[idx], momentum);
                }

            } while(errorFunc(++epochCounter, error / prototypes.size()));

            copyWeightsFromContextToNeurons();
            PerceptronType result;
            utils::for_< PerceptronType::size() - 1 >([this, &result](auto i) {
                auto& srcLayer = std::get< i.value + 1 >(m_perceptron.layers());
                auto& dstLayer = utils::get< i.value + 1 >(result.layers());
                dstLayer.setMemento(srcLayer.getMemento());
            });

            return result;
        }

      private:
        void copyWeightsToContext() {
            utils::for_< size() - 1 >([this](auto i) {
                constexpr auto idx = i.value + 1;
                auto& layer = std::get< idx >(m_perceptron.layers());
                auto& weights = std::get< idx >(m_bpContext.weights);
                auto& biases = std::get< idx >(m_bpContext.biases);
                layer.for_each([&](auto neuronIdx, auto& neuron) {
                    biases[neuronIdx.value] = neuron.getBias();
                    for (std::size_t j = 0; j < neuron.size(); ++j) {
                        weights[neuronIdx.value * neuron.size() + j] = neuron[j].weight;
                    }
                });
            });
        }

        void copyWeightsFromContextToNeurons() {
            utils::for_< size() - 1 >([this](auto i) {
                constexpr auto idx = i.value + 1;
                auto& layer = std::get< idx >(m_perceptron.layers());
                auto& weights = std::get< idx >(m_bpContext.weights);
                auto& biases = std::get< idx >(m_bpContext.biases);
                layer.for_each([&](auto neuronIdx, auto& neuron) {
                    neuron.setBias(biases[neuronIdx.value]);
                    for (std::size_t j = 0; j < neuron.size(); ++j) {
                        neuron.setWeight(j, weights[neuronIdx.value * neuron.size() + j]);
                    }
                });
            });
        }

        template< typename Iterator, typename OutputIterator >
        void forwardPass(Iterator begin, Iterator end, OutputIterator out) {
            auto& inputLayer = std::get< 0 >(m_perceptron.layers());
            unsigned int inputId = 0;
            while(begin != end) {
                for(std::size_t featureIdx = 0; featureIdx < begin->value.size();
                    ++featureIdx) {
                    inputLayer[inputId][featureIdx].weight = 1.f;
                    inputLayer[inputId].setBias({});
                    inputLayer[inputId][featureIdx].value = begin->value[featureIdx];
                }
                begin++;
                inputId++;
            }

            auto& inputLayer0 = std::get< 0 >(m_perceptron.layers());
            inputLayer0.template calculateOutputs< typename BPCtx::Forward, 0 >(m_bpContext.outputs);

            utils::for_< size() - 1U >([this](auto i) {
                auto& layer = std::get< i.value + 1 >(m_perceptron.layers());
                layer.template calculateOutputs< typename BPCtx::Forward, i.value + 1, i.value >(m_bpContext.outputs, m_bpContext);
            });

            auto& outputCtx = std::get< size() - 1U >(m_bpContext.outputs);
            for(const auto& val : outputCtx) {
                *out = val;
                ++out;
            }
        }

        /// @brief current perceptron.
        Perceptron m_perceptron;

        /// @brief the learning rate.
        Var m_leariningRate;

        /// @brief outputs stored for each step.
        std::array< Var, outputsNumber > m_outputs;

        /// @brief execution error calculator.
        ErrorCalculator< typename PerceptronType::VarType > m_errorCalculator;

        /// @brief BP context holding forward outputs reference and delta/gradient storage.
        BPCtx m_bpContext{m_perceptron.context(), {}, {}, {}, {}, {}};

        struct DummyMomentum {
            Var operator()(const Var& oldDelta, const Var& newDelta) {
                return newDelta;
            }
        };

        template< typename BPCtxT, typename MomentumFunc >
        void calculateDelta(BPCtxT& ctx, const Prototype& prototype, MomentumFunc momentum) {
            auto& layers = m_perceptron.layers();
            utils::get< size() - 1 >(layers).template calculateDeltas< BPCtxT, size() - 1 >(ctx, prototype, momentum);
            utils::for_< size() - 1 >([&layers, &ctx, &momentum](auto i) {
                constexpr auto idx = size() - i.value - 1;
                auto& frontLayer = std::get< idx >(layers);
                auto& backLayer = std::get< idx - 1 >(layers);
                backLayer.template calculateHiddenDeltas< BPCtxT, idx - 1, idx >(ctx, frontLayer, momentum);
            });
        }
    };
} // namespace nn::bp
