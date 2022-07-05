#pragma once

#include <NeuralNetwork/INeuralLayer.h>
#include <NeuralNetwork/Serialization/NeuralLayerMemento.h>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/bind/placeholders.hpp>
#include <boost/bind.hpp>

#include <range/v3/all.hpp>

#include <algorithm>
#include <array>
#include <functional>

namespace nn {

    namespace detail {
        /**
         * Represent the NeuralLayer in perceptron.
         */
        template< class NeuronType, std::size_t neuronsNumber, std::size_t inputsNumber >
        struct NeuralLayer {
            using Neuron = typename NeuronType::template adjust< inputsNumber >;
            using Var = typename Neuron::Var;
            using NeuronMemento = typename Neuron::Memento;
            using Memento = NeuralLayerMemento< NeuronMemento, neuronsNumber >;

            template< template< class > class NewType >
            using wrap =
             NeuralLayer< NewType< NeuronType >, neuronsNumber, inputsNumber >;

            template< unsigned int inputs >
            using adjust = NeuralLayer< NeuronType, neuronsNumber, inputs >;

            template< typename VarType >
            using use =
             NeuralLayer< typename NeuronType::template use< VarType >, neuronsNumber, inputsNumber >;
            static constexpr unsigned int CONST_NEURONS_NUMBER = neuronsNumber;
            static constexpr unsigned int CONST_INPUTS_NUMBER = inputsNumber;
            typedef typename std::vector< Neuron > Container;

            static_assert(neuronsNumber > 0,
                          "Invalid template argument neuronsNumber == 0");
            static_assert(inputsNumber > 0,
                          "Invalid template argument inputsNumber <= 1");

            auto cbegin() const {
                return std::cbegin(m_neurons);
            }

            auto cend() const {
                return std::cend(m_neurons);
            }

            auto begin() {
                return std::begin(m_neurons);
            }

            auto end() {
                return std::end(m_neurons);
            }

            auto size() const {
                return m_neurons.size();
            }

            const Neuron& operator[](unsigned int id) const {
                return m_neurons[id];
            }

            Neuron& operator[](unsigned int id) {
                return m_neurons[id];
            }

            void setInput(unsigned int inputId, const Var& value) {
                std::for_each(std::begin(m_neurons),
                              std::end(m_neurons),
                              std::bind(&Neuron::setInput, std::placeholders::_1, inputId, value));
            }

            const Memento getMemento() const {
                using namespace ranges;
                Memento memento;
                memento.neurons =
                 m_neurons |
                 views::transform([](const Neuron& n) { return n.getMemento(); }) |
                 ranges::to< decltype(memento.neurons) >;
                return memento;
            }

            void setMemento(const Memento& memento) {
                using namespace ranges;
                m_neurons = memento.neurons |
                            views::transform([](const NeuronMemento& m) {
                                Neuron neuron;
                                neuron.setMemento(m);
                                return neuron;
                            }) |
                            ranges::to< decltype(m_neurons) >;
            }

            Var getOutput(unsigned int outputId) const {
                return m_neurons[outputId].getOutput();
            }

            template< typename Layer >
            void calculateOutputs(Layer& nextLayer) {
                auto begin =
                 boost::make_transform_iterator(std::end(m_neurons),
                                                boost::bind(&Neuron::calcDotProduct, _1));
                auto end =
                 boost::make_transform_iterator(std::begin(m_neurons),
                                                boost::bind(&Neuron::calcDotProduct, _1));
                for(unsigned int i = 0; i < m_neurons.size(); i++) {
                    nextLayer.setInput(i, m_neurons[i].calculateOutput(begin, end));
                }
            }

            void calculateOutputs() {
                auto begin =
                 boost::make_transform_iterator(std::begin(m_neurons),
                                                boost::bind(&Neuron::calcDotProduct, ::_1));
                auto end =
                 boost::make_transform_iterator(std::end(m_neurons),
                                                boost::bind(&Neuron::calcDotProduct, ::_1));
                using IteratorType = decltype(begin);
                std::for_each(std::begin(m_neurons),
                              std::end(m_neurons),
                              std::bind(&Neuron::template calculateOutput< IteratorType >,
                                        std::placeholders::_1,
                                        begin,
                                        end));
            }

          private:
            Container m_neurons{neuronsNumber};
        };
    } // namespace detail

    template< template< template< class > class, class, std::size_t > class NeuronType,
              template< class >
              class ActivationFunctionType,
              std::size_t size,
              std::size_t inputsNumber = 2,
              typename Var = float >
    using NeuralLayer =
     detail::NeuralLayer< NeuronType< ActivationFunctionType, Var, inputsNumber >, size, inputsNumber >;
} // namespace nn
