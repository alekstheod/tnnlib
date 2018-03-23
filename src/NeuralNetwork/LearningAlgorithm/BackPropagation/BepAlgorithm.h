/**
*  Copyright (c) 2011, Alex Theodoridis
*  All rights reserved.

*  Redistribution and use in source and binary forms, with
*  or without modification, are permitted provided that the
*  following conditions are met:
*  Redistributions of source code must retain the above
*  copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above
*  copyright notice, this list of conditions and the following
*  disclaimer in the documentation and/or other materials
*  provided with the distribution.

*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
*  AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
*  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
*  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
*  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
*  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
*  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE,
*  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

#ifndef BEPTrainerH
#define BEPTrainerH
#include <NeuralNetwork/LearningAlgorithm/BackPropagation/BPNeuralLayer.h>
#include <NeuralNetwork/LearningAlgorithm/BackPropagation/ErrorFunction.h>
#include <NeuralNetwork/Perceptron/Perceptron.h>
#include <Utilities/System/Time.h>

#include <algorithm>
#include <array>
#include <boost/numeric/conversion/cast.hpp>
#include <chrono>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <queue>
#include <random>

namespace nn {

    namespace bp {

        template< typename PerceptronType, template< class > class ErrorCalculator = SquaredError >
        class BepAlgorithm {
          private:
            using Var = typename PerceptronType::Var;

            BOOST_STATIC_CONSTEXPR unsigned int inputsNumber = PerceptronType::CONST_INPUTS_NUMBER;
            BOOST_STATIC_CONSTEXPR unsigned int outputsNumber =
             PerceptronType::CONST_OUTPUTS_NUMBER;

            using Perceptron = typename PerceptronType::template wrap< BPNeuralLayer >;
            using Layers = typename Perceptron::Layers;

          public:
            using Prototype =
             typename std::tuple< std::array< Var, inputsNumber >, std::array< Var, outputsNumber > >;
            using Memento = typename Perceptron::Memento;

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
                std::get< Perceptron::CONST_LAYERS_NUMBER - 1 >(m_perceptron.layers())
                 .calculateDeltas(prototype, momentum);
                calculateDelta< Perceptron::CONST_LAYERS_NUMBER - 1 >(
                 m_perceptron.layers(), momentum);

                // Calculate weights
                utils::for_each(m_perceptron.layers(), [this](auto& layer) {
                    layer.calculateWeights(m_leariningRate);
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

                Var error = boost::numeric_cast< Var >(0.f);
                do {
                    auto seed =
                     std::chrono::system_clock::now().time_since_epoch().count();
                    std::shuffle(prototypes.begin(),
                                 prototypes.end(),
                                 std::default_random_engine(static_cast< unsigned int >(seed)));

                    const auto runTrainingStep = [&](const Var& init,
                                                     const Prototype& first) -> Var {
                        return init + executeTrainingStep(first, momentum);
                    };

                    error = std::accumulate(prototypes.begin(),
                                            prototypes.end(),
                                            boost::numeric_cast< Var >(0.f),
                                            runTrainingStep);

                } while(errorFunc(++epochCounter, error));

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

            template< unsigned int index, typename MomentumFunc >
            void calculateDelta(Layers& layers, MomentumFunc momentum) {
                std::get< index - 1 >(layers).calculateHiddenDeltas(std::get< index >(layers),
                                                                    momentum);
                if constexpr(index > 1) {
                    calculateDelta< index - 1 >(layers, momentum);
                }
            }
        };
    } // namespace bp
} // namespace nn

#endif
