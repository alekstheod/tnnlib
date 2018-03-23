#pragma once

#include <Utilities/MPL/Tuple.h>

#include <cstddef>
#include <tuple>

namespace nn {
    template< int From, int To >
    struct Range {
        bool intersect(std::size_t inputId) {
            return inputId >= From && inputId < To;
        }
    };

    template< typename Area, std::size_t NeuronId >
    struct Connection {
        Area area;
        static constexpr std::size_t neuronId = NeuronId;
    };

    template< std::size_t width, std::size_t height, std::size_t stride, std::size_t margin >
    struct ConvolutionGrid {
      private:
        static constexpr auto calcPoint(std::size_t id) {
            auto sw = width / stride;
            return (id / sw) * stride * width + (stride - 1) * width +
                   id % sw * stride + stride;
        }

        template< std::size_t areaId >
        struct Area {
            static constexpr int ar = areaId;
            static constexpr int X = areaId % width - 1;
            static constexpr int Y = areaId / width;
            bool intersect(std::size_t inputId) {
                std::size_t right = X + margin;
                std::size_t left = X - margin;
                std::size_t top = Y - margin;
                std::size_t bottom = Y + margin;

                std::size_t x = inputId % width;
                std::size_t y = inputId / width;
                return top <= y && bottom >= y && right >= x && left <= x;
            }
        };

        template< std::size_t... ints >
        static constexpr auto make(std::index_sequence< ints... >) {
            return std::tuple< Connection< Area< calcPoint(ints) >, ints >... >{};
        }

      public:
        using define = decltype(
         make(std::make_index_sequence< (width / stride) * (height / stride) >{}));
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
                    utils::for_each(m_connections, [&](auto& connection) {
                        if(neuronId == connection.neuronId &&
                           connection.area.intersect(inputId)) {
                            auto id = connection.neuronId;
                            neuron.setInput(inputId, value);
                        }
                    });

                    neuronId++;
                }
            }

          private:
            Connections m_connections;
        };
    } // namespace detail

    template< template< template< template< class > class, class, std::size_t, int > class NeuronType,
                        template< class > class ActivationFunctionType,
                        std::size_t size,
                        std::size_t inputsNumber = 2,
                        int scaleFactor = 1,
                        typename Var = float > class NeuralLayerType,
              template< template< class > class, class, std::size_t, int > class NeuronType,
              template< class > class ActivationFunctionType,
              std::size_t inputsNumber,
              typename Connections,
              typename Var = float >
    using ConvolutionLayer =
     detail::ConvolutionLayer< NeuralLayerType< NeuronType, ActivationFunctionType, std::tuple_size< Connections >::value, inputsNumber >, Connections >;
} // namespace nn
