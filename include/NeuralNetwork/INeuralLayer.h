#pragma once

#include <cstdlib>

namespace nn {

    template< typename NeuralLayerType >
    class INeuralLayer {
      public:
        using Internal = NeuralLayerType;
        using Memento = typename NeuralLayerType::Memento;
        using Var = typename Internal::Var;
        using Neuron = typename Internal::Neuron;

        static constexpr std::size_t CONST_INPUTS_NUMBER = Internal::CONST_INPUTS_NUMBER;

      private:
        Internal m_neuralLayer;

      public:
        template< typename... Args >
        INeuralLayer(Args... args) : m_neuralLayer(args...) {
        }

        INeuralLayer(const Internal& neuralLayer) : m_neuralLayer(neuralLayer) {
        }

        Internal& operator*() {
            return m_neuralLayer;
        }

        INeuralLayer(Internal neuralLayer) : m_neuralLayer(neuralLayer) {
        }

        auto cbegin() const -> decltype(m_neuralLayer.cbegin()) {
            return m_neuralLayer.begin();
        }

        auto cend() const -> decltype(m_neuralLayer.cend()) {
            return m_neuralLayer.end();
        }

        auto begin() -> decltype(m_neuralLayer.begin()) {
            return m_neuralLayer.begin();
        }

        auto end() -> decltype(m_neuralLayer.end()) {
            return m_neuralLayer.end();
        }

        auto size() const -> decltype(m_neuralLayer.size()) {
            return m_neuralLayer.size();
        }

        /**
         * @brief Will set the input value by using the given input id. If the
         * given input is is invalid value will not be set.
         * @param inputId the id of the input.
         * @param value the value which has to be set.
         */
        void setInput(std::size_t inputId, const Var& value) {
            m_neuralLayer.setInput(inputId, value);
        }

        Var getOutput(std::size_t outputId) const {
            return m_neuralLayer.getOutput(outputId);
        }

        Internal* operator->() {
            return &m_neuralLayer;
        }

        /**
         * @see {INeuralLayer}
         */
        const Neuron& operator[](std::size_t id) const {
            return m_neuralLayer[id];
        }

        Neuron& operator[](std::size_t id) {
            return m_neuralLayer[id];
        }

        operator Internal&() {
            return &m_neuralLayer;
        }


        const Memento getMemento() const {
            return m_neuralLayer.getMemento();
        }

        /**
         * @brief will set memento to the current layer.
         * @param memento instance of the POD structure which describes the
         * state.
         * @return true if succeed, false otherwise.
         */
        void setMemento(const Memento& memento) {
            m_neuralLayer.setMemento(memento);
        }

        /**
         *
         */
        template< typename Layer >
        void calculateOutputs(Layer& nextLayer) {
            m_neuralLayer.calculateOutputs(*nextLayer);
        }

        void calculateOutputs() {
            m_neuralLayer.calculateOutputs();
        }
    };

    template< typename NeuralLayer, typename Var >
    NeuralLayer createLayer(
     const typename NeuralLayer::template Memento< typename NeuralLayer::NeuronMemento, NeuralLayer::CONST_NEURONS_NUMBER >& memento) {
        NeuralLayer layer;
        layer.setMemento(memento);
        return layer;
    }
} // namespace nn
