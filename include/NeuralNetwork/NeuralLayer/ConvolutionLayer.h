#pragma once

#include <MPL/Tuple.h>

#include <cstddef>
#include <tuple>

#include <tuple>
#include <utility>

namespace nn {
    template< typename AreaType, std::size_t NeuronId >
    struct Connection {
        using Area = AreaType;
        Area area;
        static constexpr std::size_t neuronId = NeuronId;
    };

    /// The idea behind the convolution grid is that each input
    /// is checked against the areas [windows] and if it has
    /// overlap with one particular area then the input will be set.
    /// otherwise it will be dropped.each area covers a set of inputs - neurons
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
            static constexpr std::size_t ar = areaId;
            static constexpr std::size_t X = areaId % width - 1;
            static constexpr std::size_t Y = areaId / width;
            static constexpr std::size_t right = X + margin;
            static constexpr std::size_t left = X - margin;
            static constexpr std::size_t top = Y - margin;
            bool doesIntersect(std::size_t inputId) {
                std::size_t bottom = Y + margin;
                std::size_t x = inputId % width;
                std::size_t y = inputId / width;
                return top <= y && bottom >= y && right >= x && left <= x;
            }

            std::size_t localize(std::size_t inputId) {
                std::size_t x = inputId % width;
                std::size_t y = inputId / width;
                return (y - top) * (margin * 2 + 1) + x - left;
            }
        };

        template< std::size_t... ints >
        static constexpr auto makeArea(std::index_sequence< ints... >) {
            return std::tuple< Connection< Area< calcPoint(ints) >, ints >... >{};
        }

        template< typename Connections >
        struct Grid {
            static constexpr std::size_t filterWidth = margin * 2 + 1;
            static constexpr std::size_t frameSize = filterWidth * filterWidth;
            static constexpr std::size_t framesNumber =
             std::tuple_size< Connections >::value;
            static constexpr std::size_t size = width * height;
            static constexpr std::size_t rowSize = width / stride;
            Connections connections;
        };

      public:
        using define =
         Grid< decltype(makeArea(std::make_index_sequence< (width / stride) * (height / stride) >{})) >;
    };

    namespace detail {
        template< typename Internal, typename Grid >
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
            using Internal::getMemento;
            using Internal::getOutput;
            using Internal::setMemento;

            // We can't adjust this layer as the
            // number of inputs and neurons depends
            // on the convolution grid and frame sizes
            template< unsigned int inputs >
            using adjust = ConvolutionLayer;

            template< typename VarType >
            using use =
             ConvolutionLayer< typename Internal::template use< VarType >, Grid >;
            static constexpr std::size_t CONST_INPUTS_NUMBER = Grid::size;
            using Internal::CONST_NEURONS_NUMBER;

            /**
             * @see {INeuralLayer}
             */
            void setInput(unsigned int inputId, const Var& value) {
                std::size_t neuronId = 0;
                for(auto& neuron : *this) {
                    utils::for_each(m_grid.connections, [&](auto& connection) {
                        if(neuronId == connection.neuronId &&
                           connection.area.doesIntersect(inputId)) {
                            const auto localInputId = connection.area.localize(inputId);
                            neuron.setInput(localInputId, value);
                        }
                    });

                    neuronId++;
                }
            }

          private:
            Grid m_grid;
        };
    } // namespace detail

    template< template< template< template< class > class, class, std::size_t > class NeuronType, template< class > class ActivationFunctionType, std::size_t size, std::size_t inputsNumber = 2, typename Var = float >
              class NeuralLayerType,
              template< template< class > class, class, std::size_t >
              class NeuronType,
              template< class >
              class ActivationFunctionType,
              std::size_t,
              typename Grid,
              typename Var = float >
    using ConvolutionLayer =
     detail::ConvolutionLayer< NeuralLayerType< NeuronType, ActivationFunctionType, Grid::framesNumber, Grid::frameSize >, Grid >;
} // namespace nn
