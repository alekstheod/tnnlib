#pragma once

#include <MPL/Tuple.h>

#include <cstddef>
#include <tuple>

#include <tuple>
#include <utility>

#include <iostream>

namespace nn {

    namespace detail {

        template< typename T >
        constexpr std::size_t ceil(T num) {
            return (static_cast< T >(static_cast< std::size_t >(num)) == num) ?
                    static_cast< std::size_t >(num) :
                    static_cast< std::size_t >(num) + ((num > 0) ? 1 : 0);
        }

    } // namespace detail

    template< typename AreaType, std::size_t NeuronId >
    struct Connection {
        using Area = AreaType;
        Area area;
        static constexpr std::size_t neuronId = NeuronId;
    };

    template< std::size_t w, std::size_t h, std::size_t s >
    struct Kernel {
        static constexpr std::size_t width = w;
        static constexpr std::size_t height = h;
        static constexpr std::size_t stride = s;
        static constexpr std::size_t size = w * h;
    };

    /// The idea behind the convolution grid is that each input
    /// is checked against the areas [windows] and if it has
    /// overlap with one particular area then the input will be set.
    /// otherwise it will be dropped.each area covers a set of inputs - neurons
    template< std::size_t gridWidth, std::size_t gridHeight, typename Kernel >
    struct ConvolutionGrid {
        static constexpr auto calcPoint(std::size_t id) {
            auto sw = detail::ceil(static_cast< float >(gridWidth) /
                                   static_cast< float >(Kernel::stride));
            const auto lines = id / sw;
            return lines * Kernel::stride * gridWidth +
                   id * Kernel::stride % (gridWidth + 1);
        }

        template< std::size_t pos >
        struct Point {
            static constexpr std::size_t x = pos % gridWidth;
            static constexpr std::size_t y = pos / gridWidth;
        };

        template< std::size_t startPos >
        struct Area {
            static constexpr std::size_t X = startPos % gridWidth;
            static constexpr std::size_t Y = startPos / gridWidth;
            static constexpr Point< startPos > topLeft{};
            bool doesIntersect(std::size_t inputId) {
                std::size_t x = inputId % gridWidth;
                std::size_t y = inputId / gridWidth;
                return ((topLeft.y <= y) && (topLeft.y + Kernel::height > y) &&
                        (topLeft.x + Kernel::width > x) && (topLeft.x <= x));
            }

            std::size_t localize(std::size_t inputId) {
                std::size_t x = inputId % gridWidth;
                std::size_t y = inputId / gridWidth;
                return (y - topLeft.y) * Kernel::width + x - topLeft.x;
            }
        };

        template< std::size_t... ints >
        static constexpr auto makeArea(std::index_sequence< ints... >) {
            return std::tuple< Connection< Area< calcPoint(ints) >, ints >... >{};
        }

        template< typename Connections >
        struct Grid {
            using K = Kernel;
            static constexpr std::size_t framesNumber =
             std::tuple_size< Connections >::value;
            static constexpr std::size_t width = gridWidth;
            static constexpr std::size_t height = gridHeight;
            static constexpr std::size_t size = width * height;
            Connections connections;
        };

      public:
        using define = Grid< decltype(makeArea(
         std::make_index_sequence< detail::ceil((gridWidth + 1) / Kernel::stride) *
                                   detail::ceil((gridHeight + 1) / Kernel::stride) >{})) >;
    };

    namespace detail {
        template< typename Internal, typename Grid >
        class ConvolutionLayer : private Internal {
          public:
            using Var = typename Internal::Var;
            using Memento = typename Internal::Memento;
            using Internal::begin;
            using Internal::cbegin;
            using Internal::cend;
            using Internal::end;
            using Internal::size;
            using Internal::operator[];
            using Internal::calculateOutputs;
            using Internal::for_each;
            using Internal::getMemento;
            using Internal::inputs;
            using Internal::setMemento;

            // We can't adjust this layer as the
            // number of inputs and neurons depends
            // on the convolution grid and frame sizes
            template< unsigned int inputs >
            using adjust = ConvolutionLayer;

            template< typename VarType >
            using use =
             ConvolutionLayer< typename Internal::template use< VarType >, Grid >;

            void setInput(unsigned int inputId, const Var& value) {
                auto& self = *this;
                utils::for_each(m_grid.connections, [&](auto& connection) {
                    auto& neuron = self[connection.neuronId];
                    if(connection.area.doesIntersect(inputId)) {
                        const auto localInputId = connection.area.localize(inputId);
                        neuron.setInput(localInputId, value);
                    }
                });
            }

          private:
            Grid m_grid;
        };
    } // namespace detail

    template< template< template< template< class > class, class, std::size_t > class NeuronType, template< class > class ActivationFunctionType, std::size_t size, std::size_t inputsNumber = 2, typename Var = float >
              typename NeuralLayerType,
              template< template< class > class, class, std::size_t >
              typename NeuronType,
              template< class >
              class ActivationFunctionType,
              typename Grid,
              typename Var = float >
    using ConvolutionLayer =
     detail::ConvolutionLayer< NeuralLayerType< NeuronType, ActivationFunctionType, Grid::framesNumber, Grid::K::size >, Grid >;
} // namespace nn
