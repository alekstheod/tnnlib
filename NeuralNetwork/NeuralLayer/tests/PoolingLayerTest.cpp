#include "NeuralNetwork/NeuralLayer/PoolingLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/Neuron/PoolingNeuron.h"

#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch_all.hpp>

namespace {
    template< typename T >
    bool equal(T lhs, T rhs) {
        return std::abs(lhs - rhs) < std::numeric_limits< T >::epsilon();
    }

    template< typename Neuron >
    bool hasValidInputs(const Neuron& neuron, const std::vector< float >& expected) {
        for(std::size_t i = 0; i < expected.size(); i++) {
            if(!equal(neuron[i].value, expected[i])) {
                return false;
            }
        }

        return true;
    }

    SCENARIO("Pooling grid set inputs", "[layer][pooling][grid][forward]") {
        GIVEN(
         "A pooling layer for an image 6*6, kernel 3*3 and Max as pooling "
         "algorithm") {
            constexpr std::size_t width = 6;
            constexpr std::size_t height = 6;
            constexpr std::size_t stride = 3;

            using Grid =
             typename nn::ConvolutionGrid< width, height, nn::Kernel< 3, 3, stride > >::define;
            using PoolingLayer = nn::PoolingLayer< nn::NeuralLayer, nn::Max, Grid >;

            auto layer = PoolingLayer{};
            WHEN(
             "input is a grid filled with the increasing sequence of "
             "integers") {
                for(auto i : ranges::views::ints(0, 36)) {
                    layer.setInput(i, static_cast< float >(i + 1));
                }

                THEN("The layer has 4 neurons") {
                    REQUIRE(4 == layer.size());
                }

                THEN("The expected inputs can be described as a following") {
                    std::vector< std::vector< float > > inputs = {
                     {1, 2, 3, 7, 8, 9, 13, 14, 15},
                     {4, 5, 6, 10, 11, 12, 16, 17, 18},
                     {19, 20, 21, 25, 26, 27, 31, 32, 33},
                     {22, 23, 24, 28, 29, 30, 34, 35, 36}};

                    for(const auto id : ranges::views::ints(0, 4)) {
                        for(const auto input : ranges::views::ints(0, 9)) {
                            REQUIRE_THAT(layer[id][input],
                                         Catch::WithinRel(inputs[id][input]));
                        }
                    }
                }

                WHEN("calculateOutputs is called") {
                    THEN(
                     "The outputs are the max of the inputs for each neuron") {
                        layer.calculateOutputs();
                        REQUIRE_THAT(15.0f, Catch::WithinRel(layer[0].getOutput()));
                        REQUIRE_THAT(18.0f, Catch::WithinRel(layer[1].getOutput()));
                        REQUIRE_THAT(33.0f, Catch::WithinRel(layer[2].getOutput()));
                        REQUIRE_THAT(36.0f, Catch::WithinRel(layer[3].getOutput()));
                    }
                }
            }
        }
    }
} // namespace
