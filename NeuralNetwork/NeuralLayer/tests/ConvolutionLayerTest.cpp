#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"

#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

namespace {
    template< typename Neuron >
    bool hasValidInputs(const Neuron& neuron, const std::vector< float >& expected) {
        for(std::size_t i = 0; i < expected.size(); i++) {
            if(neuron[i].value != expected[i]) {
                return false;
            }
        }

        return true;
    }

    SCENARIO("Convolution layer set inputs", "[layer][convolution][forward]") {
        GIVEN(
         "A convolution layer for an image 5*5, stride = 3 and kernel 3*3") {
            constexpr std::size_t width = 5;
            constexpr std::size_t height = 5;
            constexpr std::size_t stride = 2;
            using ConvolutionGrid =
             typename nn::ConvolutionGrid< width, height, nn::Kernel< 3, 3, stride > >::define;

            using ConvolutionLayer =
             nn::ConvolutionLayer< nn::NeuralLayer, nn::Neuron, nn::SigmoidFunction, ConvolutionGrid >;
            auto layer = ConvolutionLayer{};
            WHEN(
             "input is a grid filled with the increasing sequence of "
             "integers") {
                for(auto i : ranges::views::ints(0, 25)) {
                    layer.setInput(i, static_cast< float >(i + 1));
                }

                THEN("The expected inputs can be described as a following") {
                    std::vector< std::vector< float > > inputs = {
                     {1, 2, 3, 6, 7, 8, 11, 12, 13},
                     {3, 4, 5, 8, 9, 10, 13, 14, 15},
                     {5, 0, 0, 10, 0, 0, 15, 0, 0},
                     {11, 12, 13, 16, 17, 18, 21, 22, 23},
                     {13, 14, 15, 18, 19, 20, 23, 24, 25},
                     {15, 0, 0, 20, 0, 0, 25, 0, 0},
                     {21, 22, 23, 0, 0, 0, 0, 0, 0},
                     {23, 24, 25, 0, 0, 0, 0, 0, 0},
                     {25, 0, 0, 0, 0, 0, 0, 0, 0}};
                    for(const auto id : ranges::views::ints(0, 9)) {
                        REQUIRE(hasValidInputs(layer[id], inputs[id]));
                    }
                }
            }
        }
    }

    SCENARIO("Convolution grid calculate frames",
             "[grid][convolution][forward]") {
        GIVEN(
         "A convolution layer for an image 5*5, stride = 2 and margin = "
         "1") {
            constexpr std::size_t width = 5;
            constexpr std::size_t height = 5;
            constexpr std::size_t stride = 2;
            using ConvolutionGrid =
             nn::ConvolutionGrid< width, height, nn::Kernel< 2, 2, stride > >;

            ConvolutionGrid grid;
            WHEN("calcPoint is called it") {
                grid.calcPoint(0);
                THEN("It returns a first value of the frame") {
                    REQUIRE(0 == grid.calcPoint(0));
                    REQUIRE(2 == grid.calcPoint(1));
                    REQUIRE(4 == grid.calcPoint(2));
                    REQUIRE(10 == grid.calcPoint(3));
                    REQUIRE(12 == grid.calcPoint(4));
                    REQUIRE(14 == grid.calcPoint(5));
                }
            }
        }
    }
} // namespace
