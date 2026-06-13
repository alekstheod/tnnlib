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

        static constexpr auto size() {
            return PerceptronType::size();
        }

        BepAlgorithm(Var learningRate)
         : m_leariningRate(learningRate) {
            initWeights();
        }

        template< typename MomentumFunc >
        Var executeTrainingStep(const Prototype& prototype, MomentumFunc momentum) {
            forwardPass(std::get< 0 >(prototype).begin(),
                        std::get< 0 >(prototype).end(),
                        m_outputs.begin());

            calculateDelta(m_bpContext, prototype, momentum);

            utils::for_< size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(m_layers);
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
                auto& hiddenLayer = std::get< i.value + 1 >(m_layers);
                hiddenLayer.template accumulateGradients< BPCtx, i.value + 1 >(m_bpContext);
            });

            return m_errorCalculator(m_outputs.begin(),
                                     m_outputs.end(),
                                     std::get< 1 >(prototype).begin());
        }

        void applyBatchGradients() {
            utils::for_< size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(m_layers);
                hiddenLayer.template applyGradients< BPCtx, i.value + 1 >(m_bpContext, m_leariningRate);
            });
        }

        template< typename Iterator, typename BatchErrorFunc >
        void calculateWithBatchTraining(Iterator begin,
                                        Iterator end,
                                        std::size_t batchSize,
                                        BatchErrorFunc batchErrorFunc) {
            calculateWithBatchTraining(begin, end, batchSize, batchErrorFunc, DummyMomentum());
        }

        template< typename Iterator, typename BatchErrorFunc, typename MomentumFunc >
        void calculateWithBatchTraining(Iterator begin,
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
        }

        template< typename Iterator, typename ErrorFunc >
        void calculate(Iterator begin, Iterator end, ErrorFunc func) {
            calculate(begin, end, func, DummyMomentum());
        }

        const BPCtx& context() const { return m_bpContext; }
        BPCtx& context() { return m_bpContext; }

        template< typename Iterator, typename OutputIterator >
        void evaluate(Iterator begin, Iterator end, OutputIterator out) {
            forwardPass(begin, end, out);
        }

        template< typename Iterator, typename ErrorFunc, typename MomentumFunc >
        void calculate(Iterator begin,
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
        }

      private:
        void initWeights() {
            utils::for_< size() - 1 >([this](auto i) {
                constexpr auto idx = i.value + 1;
                auto& weights = std::get< idx >(m_bpContext.weights);
                auto& biases = std::get< idx >(m_bpContext.biases);
                for (auto& w : weights) {
                    w = utils::createRandom< Var >(1) / Var{100};
                }
                for (auto& b : biases) {
                    b = utils::createRandom< Var >(1);
                }
            });
        }

        template< typename Iterator, typename OutputIterator >
        void forwardPass(Iterator begin, Iterator end, OutputIterator out) {
            auto& inputLayer = std::get< 0 >(m_layers);
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

            auto& inputLayer0 = std::get< 0 >(m_layers);
            inputLayer0.template calculateOutputs< typename BPCtx::Forward, 0 >(m_bpContext.outputs);

            utils::for_< size() - 1U >([this](auto i) {
                auto& layer = std::get< i.value + 1 >(m_layers);
                layer.template calculateOutputs< typename BPCtx::Forward, i.value + 1, i.value >(m_bpContext.outputs, m_bpContext);
            });

            auto& outputCtx = std::get< size() - 1U >(m_bpContext.outputs);
            for(const auto& val : outputCtx) {
                *out = val;
                ++out;
            }
        }

        Layers m_layers{};
        Var m_leariningRate;
        std::array< Var, outputsNumber > m_outputs;
        ErrorCalculator< typename PerceptronType::VarType > m_errorCalculator;
        BPCtx m_bpContext{};

        struct DummyMomentum {
            Var operator()(const Var& oldDelta, const Var& newDelta) {
                return newDelta;
            }
        };

        template< typename BPCtxT, typename MomentumFunc >
        void calculateDelta(BPCtxT& ctx, const Prototype& prototype, MomentumFunc momentum) {
            auto& layers = m_layers;
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
