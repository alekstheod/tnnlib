#pragma once

#include <NeuralNetwork/Neuron/INeuron.h>
#include <NeuralNetwork/Neuron/Input.h>
#include <NeuralNetwork/Serialization/NeuronMemento.h>


#include <System/Time.h>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <utility>

namespace nn {
    /**
     * Neuron class.
     * Represent the neuron in the Neural layer.
     * Contains equation for output value calculation
     * and list of accepted inputs.
     */
    namespace detail {
        template< typename OutputFunctionType, std::size_t inputsNumber >
        struct Neuron : INeuron< typename OutputFunctionType::Var > {
            using OutputFunction = OutputFunctionType;
            using Var = typename OutputFunction::Var;
            using Memento = NeuronMemento< Var, inputsNumber >;
            using Input = nn::Input< Var >;

            /// @brief a list of the inputs first is the weight, second is the
            /// value
            using Inputs = typename Memento::Inputs;

            template< typename VarType >
            using use =
             Neuron< typename OutputFunctionType::template use< VarType >, inputsNumber >;

            template< std::size_t inputs >
            using adjust = Neuron< OutputFunctionType, inputs >;

            static_assert(inputsNumber > 0, "Invalid number of inputs");

            auto cbegin() const {
                return std::cbegin(m_inputs);
            }

            auto cend() const {
                return std::cend(m_inputs);
            }

            static constexpr auto size() {
                return inputsNumber;
            }

            Input& operator[](std::size_t id) {
                return m_inputs[id];
            }

            const Input& operator[](const ::size_t id) const {
                return m_inputs[id];
            }

            void setWeight(std::size_t weightId, const Var& weight) {
                m_inputs[weightId].weight = weight;
            }

            const Var& getBias() const {
                return m_bias;
            }

            const Var& getWeight(std::size_t weightId) const {
                return m_inputs[weightId].weight;
            }

            const Memento getMemento() const {
                return Memento{m_bias, m_inputs};
            }

            void setMemento(const Memento& memento) {
                const auto& inputs = memento.inputs;
                std::copy(inputs.begin(), inputs.end(), m_inputs.begin());
                m_bias = memento.bias;
            }

            void setInput(unsigned int inputId, const Var& value) {
                m_inputs[inputId].value = value;
            }

            Var calcDotProduct() const {
                const auto calcInput = [](const Input& input) {
                    return input.value * input.weight;
                };

                auto begin = boost::make_transform_iterator(m_inputs.cbegin(), calcInput);
                auto end = boost::make_transform_iterator(m_inputs.cend(), calcInput);
                return m_activationFunction.sum(begin, end, m_bias);
            }

            const Var& getOutput() const {
                return m_output;
            }

            unsigned int getInputsNumber() const {
                return m_inputs.size();
            }

            void setBias(Var weight) {
                m_bias = weight;
            }

            const Var& getNeuronWeight() const {
                return m_bias;
            }

            template< typename Iterator >
            const Var& calculateOutput(Iterator begin, Iterator end) {
                m_output = m_activationFunction.calculate(calcDotProduct(), begin, end);
                return m_output;
            }

            template< typename Test >
            void supportTest(Test&);

          private:
            /**
             * @brief Instance of output calculation equation.
             * @brief The equation should be provided by implementation of
             * IEquationFactory interface.
             */
            OutputFunction m_activationFunction;

            /**
             * @brief List of neurons inputs.
             */
            Inputs m_inputs{};

            /**
             * @brief Neurons weight.
             * @brief Needed in order to improve the flexibility of neural
             * network.
             */
            Var m_bias{utils::createRandom< Var >(1)};

            /**
             * @brief The neurons output.
             * @brief The output will be calculated with using the instance of
             * calculation equation.
             */
            Var m_output{};

            /**
             * @brief Needed in order to calculate the neurons output value.
             */
            Var m_sum{};
        };
    } // namespace detail

    template< template< class > class OutputFunctionType, typename VarType = float, std::size_t inputsNumber = 1 >
    using Neuron = detail::Neuron< OutputFunctionType< VarType >, inputsNumber >;
} // namespace nn
