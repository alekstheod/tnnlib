#pragma once

#include <Utilities/MPL/Tuple.h>

#include <cstddef>
#include <tuple>

namespace nn {
    template< int From, int To >
    struct Range {
        bool contains(std::size_t inputId) {
            return inputId >= From && inputId < To;
        }
    };

    template< typename Area, std::size_t NeuronId >
    struct Connection {
        Area area;
        static constexpr std::size_t neuronId = NeuronId;
    };

    namespace detail {
        template< typename Internal, typename Connections >
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

            template< template< class > class NewType >
            using wrap =
             ConvolutionLayer< typename Internal::template wrap< NewType >, Connections >;

            template< unsigned int inputs >
            using resize =
             ConvolutionLayer< typename Internal::template resize< inputs >, Connections >;

            template< typename VarType >
            using use =
             ConvolutionLayer< typename Internal::template use< VarType >, Connections >;
            using Internal::CONST_INPUTS_NUMBER;
            using Internal::CONST_NEURONS_NUMBER;

            /**
             * @see {INeuralLayer}
             */
            void setInput(unsigned int inputId, const Var& value) {
                std::size_t neuronId = 0;
                for(auto& neuron : *this) {
                    utils::for_each(m_connections, [&](auto connection) {
                        if(connection.area.contains(inputId) &&
                           neuronId == connection.neuronId) {
                            neuron.setInput(inputId, value);
                        }
                    });
                }

                neuronId++;
            }

          private:
            Connections m_connections;
        };
    } // namespace detail

    template< typename Internal, typename... Connections >
    using ConvolutionLayer =
     detail::ConvolutionLayer< Internal, std::tuple< Connections... > >;
} // namespace nn
