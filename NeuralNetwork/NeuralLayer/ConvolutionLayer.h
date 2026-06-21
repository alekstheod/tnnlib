#pragma once

#include <MPL/Tuple.h>

#include <cstddef>
#include <tuple>
#include <utility>

namespace nn {

    template< std::size_t w, std::size_t h, std::size_t s >
    struct Kernel {
        static constexpr std::size_t width = w;
        static constexpr std::size_t height = h;
        static constexpr std::size_t stride = s;
        static constexpr std::size_t size = w * h;
    };

    template< std::size_t Width, std::size_t Height, typename Kernel_ >
    struct SlidingWindow {
        using K = Kernel_;
        static constexpr std::size_t gridWidth = Width;
        static constexpr std::size_t gridHeight = Height;
        static constexpr std::size_t windowsPerRow =
            (gridWidth + K::stride - 1) / K::stride;
        static constexpr std::size_t windowsPerCol =
            (gridHeight + K::stride - 1) / K::stride;
        static constexpr std::size_t framesNumber = windowsPerRow * windowsPerCol;
        static constexpr std::size_t width = gridWidth;
        static constexpr std::size_t height = gridHeight;
        static constexpr std::size_t size = width * height;

        static bool contains(std::size_t inputId, std::size_t windowId) {
            const std::size_t winX = (windowId % windowsPerRow) * K::stride;
            const std::size_t winY = (windowId / windowsPerRow) * K::stride;
            const std::size_t inX = inputId % gridWidth;
            const std::size_t inY = inputId / gridWidth;
            return (inX >= winX && inX < winX + K::width &&
                    inY >= winY && inY < winY + K::height);
        }

        static std::size_t localize(std::size_t inputId, std::size_t windowId) {
            const std::size_t winX = (windowId % windowsPerRow) * K::stride;
            const std::size_t winY = (windowId / windowsPerRow) * K::stride;
            const std::size_t inX = inputId % gridWidth;
            const std::size_t inY = inputId / gridWidth;
            return (inY - winY) * K::width + (inX - winX);
        }
    };

    namespace detail {
        template< typename Internal, typename Grid >
        class ConvolutionLayer : private Internal {
          public:
            using Var = typename Internal::Var;
            using ActivationFunctions = typename Internal::ActivationFunctions;
            using Internal::begin;
            using Internal::cbegin;
            using Internal::cend;
            using Internal::end;
            using Internal::size;
            using Internal::operator[];
            using Internal::for_each;
            using Internal::inputs;

            // We can't adjust this layer as the
            // number of inputs and neurons depends
            // on the convolution grid and frame sizes
            template< unsigned int inputs >
            using adjust = ConvolutionLayer;

            template< typename VarType >
            using use =
             ConvolutionLayer< typename Internal::template use< VarType >, Grid >;

            template< template< class > typename NewType >
            using wrap =
             ConvolutionLayer< typename Internal::template wrap< NewType >, Grid >;

            void setInput(unsigned int inputId, const Var& value) {
                auto& self = *this;
                for(std::size_t w = 0; w < Grid::framesNumber; ++w) {
                    auto& neuron = self[w];
                    if(Grid::contains(inputId, w)) {
                        neuron.setInput(Grid::localize(inputId, w), value);
                    }
                }
            }

            template< typename Context, std::size_t myIdx >
            void calculateOutputs(Context& ctx) {
                auto& myOutputs = std::get< myIdx >(ctx);
                std::array< Var, size() > dotProducts;
                Internal::for_each([&](auto i, auto& neuron) {
                    dotProducts[i.value] = neuron.calcDotProduct();
                });
                Internal::for_each([&](auto i, auto& neuron) {
                    const auto output = neuron.calculateOutput(
                     dotProducts[i.value], std::cbegin(dotProducts), std::cend(dotProducts));
                    myOutputs[i.value] = output;
                });
            }

            template< typename Context, std::size_t myIdx, std::size_t predecessorIdx >
            void calculateOutputs(Context& ctx) {
                auto& predecessorOutputs = std::get< predecessorIdx >(ctx);
                auto& myOutputs = std::get< myIdx >(ctx);
                for (std::size_t i = 0; i < predecessorOutputs.size(); ++i) {
                    setInput(i, predecessorOutputs[i]);
                }
                std::array< Var, size() > dotProducts;
                Internal::for_each([&](auto i, auto& neuron) {
                    dotProducts[i.value] = neuron.calcDotProduct();
                });
                Internal::for_each([&](auto i, auto& neuron) {
                    const auto output = neuron.calculateOutput(
                     dotProducts[i.value], std::cbegin(dotProducts), std::cend(dotProducts));
                    myOutputs[i.value] = output;
                });
            }

            template< typename Context, std::size_t myIdx, std::size_t predecessorIdx, typename W >
            void calculateOutputs(Context& ctx, const W& wctx) {
                auto& predecessorOutputs = std::get< predecessorIdx >(ctx);
                auto& myOutputs = std::get< myIdx >(ctx);
                const auto& weights = std::get< myIdx >(wctx.weights);
                const auto& biases = std::get< myIdx >(wctx.biases);
                for (std::size_t i = 0; i < predecessorOutputs.size(); ++i) {
                    setInput(i, predecessorOutputs[i]);
                }
                constexpr auto neuronInputs = Internal::inputs();
                std::array< Var, size() > dotProducts;
                auto& self = *this;
                Internal::for_each([&](auto i, auto&) {
                    Var dot = biases[i.value];
                    for (std::size_t j = 0; j < neuronInputs; ++j) {
                        dot += self[i.value][j].value * weights[i.value * neuronInputs + j];
                    }
                    dotProducts[i.value] = dot;
                });
                Internal::for_each([&](auto i, auto& neuron) {
                    const auto output = neuron.calculateOutput(
                     dotProducts[i.value], std::cbegin(dotProducts), std::cend(dotProducts));
                    myOutputs[i.value] = output;
                });
            }
        };
    } // namespace detail

    template< template< template< template< class > class, class, std::size_t > class NeuronType,
                        template< class > class ActivationFunctionType,
                        std::size_t size,
                        std::size_t inputsNumber = 2,
                        typename Var = float > typename NeuralLayerType,
              template< template< class > class, class, std::size_t > typename NeuronType,
              template< class > class ActivationFunctionType,
              typename Grid,
              typename Var = float >
    using ConvolutionLayer =
     detail::ConvolutionLayer< NeuralLayerType< NeuronType, ActivationFunctionType, Grid::framesNumber, Grid::K::size >, Grid >;

} // namespace nn
