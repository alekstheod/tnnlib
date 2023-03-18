#pragma once

#include "NeuralNetwork/Serialization/ComplexLayerMemento.h"

namespace nn {

    /// @brief adapter of perceptron to a neural layer.
    template< typename Perceptron >
    class ComplexLayer {
      public:
        using Var = typename Perceptron::Var;
        using Memento = ComplexLayerMemento< Var >;
        using OutputLayerType = typename Perceptron::OutputLayerType;
        using InputLayerType = typename Perceptron::InputLayerType;
        static constexpr std::size_t CONST_LAYERS_NUMBER = Perceptron::CONST_LAYERS_NUMBER;
        static constexpr std::size_t CONST_NEURONS_NUMBER = OutputLayerType::CONST_NEURONS_NUMBER;
        static constexpr std::size_t CONST_INPUTS_NUMBER = InputLayerType::CONST_INPUTS_NUMBER;

        /// @brief will rebind the variable type for an underlying perceptron.
        template< typename VarType >
        using use = ComplexLayer< typename Perceptron::template use< VarType > >;

        /// @brief will rebind the number of inputs for an underlying perceptron
        template< std::size_t inputs >
        using resize = ComplexLayer< typename Perceptron::template resize< inputs > >;

      private:
        std::array< Var, CONST_INPUTS_NUMBER > m_inputs;
        std::array< Var, CONST_NEURONS_NUMBER > m_outputs;
        Perceptron m_perceptron;

      public:
        ComplexLayer() {
        }

        ComplexLayer(const Perceptron& perceptron) : m_perceptron(perceptron) {
        }

        /**
         * @see {INeuralLayer}
         */
        Memento getMemento() {
            return m_perceptron.getMemento();
        }

        /**
         * @see {INeuralLayer}
         */
        void setMemento(const Memento& memento) {
            m_perceptron.setMemento();
        }

        /**
         * @see {INeuralLayer}
         */
        template< typename Layer >
        void calculateOutputs(Layer& nextLayer) {
            m_perceptron.calculate(m_inputs.begin(), m_inputs.end(), m_outputs.begin());
            for(std::size_t i = 0; i < m_outputs.size(); i++) {
                nextLayer.setInput(i, m_outputs[i]);
            }
        }

        /**
         * @see {INeuralLayer}
         */
        void calculateOutputs() {
            m_perceptron.calculate(m_inputs.begin(), m_inputs.end(), m_outputs.begin());
        }

        /**
         * @see {INeuralLayer}
         */
        auto begin() const {
            return std::get< CONST_LAYERS_NUMBER - 1 >(m_perceptron.layers()).begin();
        }

        /**
         * @see {INeuralLayer}
         */
        auto end() const {
            return std::get< CONST_LAYERS_NUMBER - 1 >(m_perceptron.layers()).end();
        }

        /**
         * @see {INeuralLayer}
         */
        void setInput(std::size_t inputId, const Var& value) {
            m_inputs[inputId] = value;
        }
    };
} // namespace nn
