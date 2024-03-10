#pragma once

#include "BepAlgorithm.h"
#include "NeuralNetwork/BackPropagation/ErrorFunction.h"

#include <range/v3/view/subrange.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <chrono>

namespace nn::bp {

    template< typename PerceptronType, template< class > class ErrorCalculator = SquaredError >
    struct BpttAlgorithm : private BepAlgorithm< PerceptronType, SquaredError > {
        using Base = BepAlgorithm< PerceptronType, SquaredError >;
        using typename Base::DummyMomentum;
        using typename Base::Input;
        using typename Base::Layers;
        using typename Base::Memento;
        using typename Base::Var;

        using Prototype =
         typename std::tuple< std::vector< std::array< Input, PerceptronType::inputs() > >,
                              std::array< Var, PerceptronType::outputs() > >;

        BpttAlgorithm(Var learningRate) : Base(learningRate) {
        }

        template< typename MomentumFunc >
        Var executeTrainingStep(const Prototype& prototype, MomentumFunc momentum) {
            auto& lastLayer = std::get< Base::size() - 1 >(Base::m_perceptron.layers());
            std::array< Var, lastLayer.size() > deltas;
            for(const auto& intputs : std::get< 0 >(prototype)) {
                // Calculate values with current inputs
                Base::m_perceptron.calculate(intputs.begin(),
                                             intputs.end(),
                                             Base::m_outputs.begin());

                // Calculate deltas
                lastLayer.calculateDeltas(prototype, momentum);
                lastLayer.for_each([&](auto i, auto& neuron) {
                    deltas[i.value] +=
                     neuron.calculateDelta(std::get< 1 >(prototype)[i.value], momentum);
                });
            }

            lastLayer.for_each(
             [&](auto i, auto& neuron) { neuron.setDelta(deltas[i.value]); });

            // Calculate weights
            utils::for_< Base::size() - 1 >([this](auto i) {
                auto& hiddenLayer = std::get< i.value + 1 >(Base::m_perceptron.layers());
                hiddenLayer.calculateWeights(Base::m_leariningRate);
            });

            Var error{};
            for(const auto& inputs : std::get< 0 >(prototype)) {
                Base::m_perceptron.calculate(inputs.begin(),
                                             inputs.end(),
                                             Base::m_outputs.begin());
                // Calculate error
                error += Base::m_errorCalculator(Base::m_outputs.begin(),
                                                 Base::m_outputs.end(),
                                                 std::get< 1 >(prototype).begin());
            }

            return error;
        }


        template< typename ErrorFunc, typename MomentumFunc = DummyMomentum >
        PerceptronType calculate(std::vector< Prototype > prototypes,
                                 ErrorFunc errorFunc,
                                 MomentumFunc momentum = {}) {
            unsigned int epochCounter = 0;

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
            perceptron.setMemento(Base::m_perceptron.getMemento());

            return perceptron;
        }
    };
} // namespace nn::bp
