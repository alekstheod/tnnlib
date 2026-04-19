#pragma once

#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/BPConvolutionNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/ErrorFunction.h"
#include <System/Time.h>

#include <algorithm>
#include <array>
#include <chrono>

namespace nn::bp {

    template< typename PerceptronType,
              template< typename, size_t > class OptimizerType,
              template< class > class ErrorCalculator = SquaredError >
    class BepAlgorithm {
        using Var = typename PerceptronType::Var;
        using Input = typename PerceptronType::Input;

        static constexpr unsigned int inputsNumber = PerceptronType::inputs();
        static constexpr unsigned int outputsNumber = PerceptronType::outputs();

        template< typename Layer >
        using BPWrappedLayer = BPNeuralLayer< Layer, OptimizerType >;

        using Perceptron = typename PerceptronType::template wrap< BPWrappedLayer >;
        using Layers = typename Perceptron::Layers;

      public:
        using Prototype =
         typename std::tuple< std::array< Input, inputsNumber >, std::array< Var, outputsNumber > >;
        using Memento = typename Perceptron::Memento;


        static constexpr auto size() {
            return PerceptronType::size();
        }

        /// @brief execution of the single learning step in this algorithm.
        /// @param prototype a prototype used for this step.
        /// @param momentum a callback which will calculate a new delta,
        /// used in order to introduce momentum.
        /// @return error on this step.
        template< typename MomentumFunc >
        Var executeTrainingStep(const Prototype& prototype, MomentumFunc momentum) {
            m_perceptron.calculate(std::get< 0 >(prototype).begin(),
                                   std::get< 0 >(prototype).end(),
m_outputs.begin());

            calculateDelta(prototype, momentum);

            // Calculate weights for dense layers (they have internal optimizer)
            utils::for_< size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(m_perceptron.layers());
                hiddenLayer.calculateWeights();
            });

            return m_errorCalculator(m_outputs.begin(),
                                     m_outputs.end(),
                                     std::get< 1 >(prototype).begin());
        }

        template< typename MomentumFunc >
        Var executeBatchTrainingStep(const Prototype& prototype, MomentumFunc momentum) {
            m_perceptron.calculate(std::get< 0 >(prototype).begin(),
                                   std::get< 0 >(prototype).end(),
                                   m_outputs.begin());

            calculateDelta(prototype, momentum);

            utils::for_< size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(m_perceptron.layers());
                hiddenLayer.accumulateGradients();
            });

            // Apply gradients
            utils::for_< size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(m_perceptron.layers());
                hiddenLayer.applyGradients();
            });

            return m_errorCalculator(m_outputs.begin(),
                                     m_outputs.end(),
                                     std::get< 1 >(prototype).begin());
        }

        void applyBatchGradients() {
            utils::for_< size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(m_perceptron.layers());
                hiddenLayer.applyGradients();
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

            PerceptronType result;
            utils::for_< PerceptronType::size() - 1 >([this, &result](auto i) {
                auto& srcLayer = std::get< i.value + 1 >(m_perceptron.layers());
                auto& dstLayer = utils::get< i.value + 1 >(result.layers());
                dstLayer.setMemento(srcLayer.getMemento());
            });

            return result;
        }

      private:
        /// @brief current perceptron.
        Perceptron m_perceptron;

        /// @brief outputs stored for each step.
        std::array< Var, outputsNumber > m_outputs;

        /// @brief execution error calculator.
        ErrorCalculator< typename PerceptronType::Var > m_errorCalculator;

        struct DummyMomentum {
            Var operator()(const Var& oldDelta, const Var& newDelta) {
                return newDelta;
            }
        };

        template< typename MomentumFunc >
        void calculateDelta(const Prototype& prototype, MomentumFunc momentum) {
            auto& layers = m_perceptron.layers();
            utils::get< size() - 1 >(layers).calculateDeltas(prototype, momentum);
            utils::for_< size() - 1 >([&layers, &momentum](auto i) {
                constexpr auto idx = size() - i.value - 1;
                auto& frontLayer = std::get< idx >(layers);
                auto& backLayer = std::get< idx - 1 >(layers);
                backLayer.calculateHiddenDeltas(frontLayer, momentum);
            });
        }
    };
} // namespace nn::bp
