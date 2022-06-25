#pragma once

#include <cstdlib>

namespace nn {

    template< class Neuron >
    class INeuron {
      public:
        using Var = typename Neuron::Var;
        using OutputFunction = typename Neuron::OutputFunction;
        using Memento = typename Neuron::Memento;
        using Input = typename Neuron::Input;
        using NeuronType = Neuron;

        /**
         * @brief will calculate the output of the neuron.
         * @return the calculated value.
         */
        template< typename Iterator >
        const Var& calculateOutput(Iterator begin, Iterator end) {
            return m_neuron.calculateOutput(begin, end);
        }

        /**
         * @brief Will return the number of inputs for current neuron.
         * @return the number of inputs.
         */
        std::size_t size() const {
            return m_neuron.size();
        }

        Input& operator[](std::size_t id) {
            return m_neuron[id];
        }

        const Input& operator[](std::size_t id) const {
            return m_neuron[id];
        }

        /**
         * @brief will return the output value of the current neuron.
         * @return a current output value of the neuron.
         */
        Var getOutput() const {
            return m_neuron.getOutput();
        }

        Var calcDotProduct() const {
            return m_neuron.calcDotProduct();
        }

        /**
         * @brief set the value to the given input
         * @param inputId the id of the input in which the value will be
         * assigned.
         * @param value a value.
         */
        void setInput(unsigned int inputId, const Var& value) {
            m_neuron.setInput(inputId, value);
        }

        /**
         *	Will set the inputs weight value.
         *	@param weightId the inputs identifier to set the new weight value.
         *	@param weigh a new weight value.
         */
        void setWeight(unsigned int weightId, const Var& weight) {
            m_neuron.setWeight(weightId, weight);
        }

        const Var& getBias() const {
            return m_neuron.getBias();
        }

        void setBias(Var weight) {
            m_neuron.setBias(weight);
        }

        const Var& getWeight(unsigned int weightId) const {
            return m_neuron.getWeight(weightId);
        }

        const Memento getMemento() const {
            return m_neuron.getMemento();
        }

        Neuron& operator*() {
            return m_neuron;
        }

        const Neuron& operator*() const {
            return m_neuron;
        }

        const Neuron* operator->() const {
            return &m_neuron;
        }

        Neuron* operator->() {
            return &m_neuron;
        }

      private:
        Neuron m_neuron;
    };
} // namespace nn
