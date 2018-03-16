#pragma once

#include <Utilities/MPL/Tuple.h>

#include <cstddef>
#include <tuple>

namespace nn {
    template< std::size_t From, std::size_t To >
    struct InputRange {
        static constexpr std::size_t from = From;
        static constexpr std::size_t to = To;
    };

    template< typename Range, std::size_t NeuronId >
    struct Connection {
        Range inputRange;
        static constexpr std::size_t neuronId = NeuronId;
    };

    template< typename Internal, typename... Connection >
    class ConvolutionLayer : private Internal {
      public:
        using Neuron = typename Internal::Neuron;
        using Var = typename Internal::Var;
        using Memento = typename Internal::Memento;
        using Internal::begin;
        using Internal::cbegin;
        using Internal::cend;
        using Internal::end;
        using Internal::size;
        using Internal::operator[];
        using Internal::calculateOutputs;
        using Internal::getBias;
        using Internal::getInputWeight;
        using Internal::getMemento;
        using Internal::getOutput;
        using Internal::setInput;
        using Internal::setMemento;
        using typename Internal::Neuron;

        template< template< class > class NewType >
        using wrap =
         ConvolutionLayer< typename Internal::template wrap< NewType >, Connection... >;

        template< unsigned int inputs >
        using resize =
         ConvolutionLayer< typename Internal::template resize< inputs >, Connection... >;

        template< typename VarType >
        using use =
         ConvolutionLayer< typename Internal::template use< VarType >, Connection... >;
        using Internal::CONST_INPUTS_NUMBER;
        using Internal::CONST_NEURONS_NUMBER;

        /**
         * @see {INeuralLayer}
         */
        void setInput(unsigned int inputId, const Var& value) {
            std::size_t neuronId = 0;
            for(auto& neuron : *this) {
                utils::for_each(m_connections, [&](auto connection) {
                    if(inputId >= connection.inputRange.from &&
                       inputId <= connection.inputRange.to) {
                        neuron.setInput(inputId, value);
                    }
                });

                neuronId++;
            }
        }

      private:
        std::tuple< Connection... > m_connections;
    };
} // namespace nn
