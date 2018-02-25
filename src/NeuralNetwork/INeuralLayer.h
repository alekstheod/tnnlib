#ifndef INEURALLAYER_H
#define INEURALLAYER_H

#include <boost/config.hpp>
#include <cstdlib>

namespace nn {

    template< typename NeuralLayerType >
    class INeuralLayer {
      public:
        typedef NeuralLayerType NeuralLayer;
        typedef typename NeuralLayerType::Memento Memento;
        typedef typename NeuralLayer::Var Var;
        typedef typename NeuralLayer::Neuron Neuron;
        typedef typename NeuralLayer::const_iterator const_iterator;
        typedef typename NeuralLayer::iterator iterator;
        typedef typename NeuralLayer::reverse_iterator reverse_iterator;
        typedef typename NeuralLayer::const_reverse_iterator const_reverse_iterator;
        BOOST_STATIC_CONSTEXPR std::size_t CONST_INPUTS_NUMBER = NeuralLayer::CONST_INPUTS_NUMBER;

      private:
        NeuralLayer m_neuralLayer;

      public:
        template< typename... Args >
        INeuralLayer(Args... args) : m_neuralLayer(args...) {
        }

        INeuralLayer(const NeuralLayer& neuralLayer)
         : m_neuralLayer(neuralLayer) {
        }

        NeuralLayer& operator*() {
            return m_neuralLayer;
        }

        INeuralLayer(NeuralLayer neuralLayer) : m_neuralLayer(neuralLayer) {
        }

        const_iterator find(std::size_t neuronId) const {
            return m_neuralLayer.find(neuronId);
        }

        const_iterator begin() const {
            return m_neuralLayer.begin();
        }

        const_iterator end() const {
            return m_neuralLayer.end();
        }

        iterator begin() {
            return m_neuralLayer.begin();
        }

        iterator end() {
            return m_neuralLayer.end();
        }

        reverse_iterator rbegin() {
            return m_neuralLayer.rbegin();
        }

        reverse_iterator rend() {
            return m_neuralLayer.rend();
        }

        const_reverse_iterator rbegin() const {
            return m_neuralLayer.rbegin();
        }

        const_reverse_iterator rend() const {
            return m_neuralLayer.rend();
        }

        std::size_t size() const {
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

        NeuralLayer* operator->() {
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

        operator NeuralLayerType&() {
            return &m_neuralLayer;
        }

        const Var& getBias(std::size_t neuronId) const {
            return m_neuralLayer.getBias(neuronId);
        }

        const Var& getInputWeight(unsigned int neuronId, unsigned int weightId) const {
            return m_neuralLayer.getInputWeight(neuronId, weightId);
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

        ~INeuralLayer() {
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

#endif // INEURALLAYER_H
