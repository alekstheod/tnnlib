#pragma once

#include "NeuralNetwork/Serialization/NeuronMemento.h"

#include <algorithm>
#include <numeric>

namespace nn {
    template< typename T >
    struct Avg {
        template< typename I >
        T operator()(I begin, I end) const {
            T sum = std::accumulate(begin, end, T{});
            return sum / std::distance(begin, end);
        }
    };

    template< typename T >
    struct Max {
        template< typename I >
        T operator()(I begin, I end) const {
            return *std::max_element(begin, end);
        }
    };

    namespace detail {

        struct EmptyMemento {};

        template< typename VarType, typename PoolingAlgo, std::size_t inputsNumber >
        struct PoolingNeuron {
            using Var = VarType;

            using OutputFunction = PoolingAlgo;
            using Memento = StaticNeuronMemento;
            using Input = Var;
            using Inputs = std::array< Var, inputsNumber >;

            template< typename V >
            using use = PoolingNeuron< V, PoolingAlgo, inputsNumber >;

            template< std::size_t inputs >
            using resize = PoolingNeuron< Var, PoolingAlgo, inputs >;

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

            Var& operator[](std::size_t id) {
                return m_inputs[id];
            }

            const Var& operator[](const ::size_t id) const {
                return m_inputs[id];
            }

            const Memento getMemento() const {
                return {};
            }

            void setMemento(const Memento& memento) {
            }

            void setInput(std::size_t inputId, const Var& value) {
                m_inputs[inputId] = value;
            }

            const Var& getOutput() const {
                return m_output;
            }

            Var getWeight(std::size_t) const {
                return {};
            }

            Var getBias() const {
                return {};
            }

            template< typename Iterator >
            const Var& calculateOutput(const Var&, Iterator, Iterator) {
                m_output = PoolingAlgo{}(std::begin(m_inputs), std::end(m_inputs));
                return m_output;
            }

            Var calcDotProduct() const {
                return Var{};
            }

          private:
            std::array< Var, inputsNumber > m_inputs{Var{}};
            Var m_output{};
        };
    } // namespace detail

    template< template< class > class PoolingAlgo = nn::Avg, typename VarType = float, std::size_t inputsNumber = 1 >
    using PoolingNeuron =
     detail::PoolingNeuron< VarType, PoolingAlgo< VarType >, inputsNumber >;
} // namespace nn
