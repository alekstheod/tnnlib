#pragma once

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
        using Var = typename PerceptronType::Var;
        using Input = typename PerceptronType::Input;

        static constexpr unsigned int inputsNumber = PerceptronType::inputs();
        static constexpr unsigned int outputsNumber = PerceptronType::outputs();

        using Perceptron = typename PerceptronType::template wrap< BPNeuralLayer >;
        using Layers = typename Perceptron::Layers;

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
        BepAlgorithm(Var learningRate) : m_leariningRate(learningRate) {
        }

        /// @brief execution of the single learning step in this algorithm.
        /// @param prototype a prototype used for this step.
        /// @param momentum a callback which will calculate a new delta,
        /// used in order to introduce momentum.
        /// @return error on this step.
        template< typename MomentumFunc >
        Var executeTrainingStep(const Prototype& prototype, MomentumFunc momentum) {
            // Calculate values with current inputs
            m_perceptron.calculate(std::get< 0 >(prototype).begin(),
                                   std::get< 0 >(prototype).end(),
                                   m_outputs.begin());

            // Calculate deltas
            std::get< Perceptron::size() - 1 >(m_perceptron.layers()).calculateDeltas(prototype, momentum);
            calculateDelta(prototype, momentum);

            // Calculate weights
            utils::for_< size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(m_perceptron.layers());
                hiddenLayer.calculateWeights(m_leariningRate);
            });

            m_perceptron.calculate(std::get< 0 >(prototype).begin(),
                                   std::get< 0 >(prototype).end(),
                                   m_outputs.begin());

            // Calculate error
            return m_errorCalculator(m_outputs.begin(),
                                     m_outputs.end(),
                                     std::get< 1 >(prototype).begin());
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

            PerceptronType perceptron;
            perceptron.setMemento(m_perceptron.getMemento());

            return perceptron;
        }

      private:
        /// @brief current perceptron.
        Perceptron m_perceptron;

        /// @brief the learning rate.
        Var m_leariningRate;

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
