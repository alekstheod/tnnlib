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
                                         Catch::Matchers::WithinRel(inputs[id][input]));
                        }
                    }
                }

                WHEN("calculateOutputs is called") {
                    THEN(
                     "The outputs are the max of the inputs for each neuron") {
                        layer.calculateOutputs();
                        REQUIRE_THAT(15.0f, Catch::Matchers::WithinRel(layer[0].getOutput()));
                        REQUIRE_THAT(18.0f, Catch::Matchers::WithinRel(layer[1].getOutput()));
                        REQUIRE_THAT(33.0f, Catch::Matchers::WithinRel(layer[2].getOutput()));
                        REQUIRE_THAT(36.0f, Catch::Matchers::WithinRel(layer[3].getOutput()));
                    }
                }
            }
        }
    }

    SCENARIO("Pooling with Average algorithm", "[layer][pooling][average]") {
        GIVEN(
         "A pooling layer for an image 4*4, kernel 2*2 stride 2 with Avg "
         "pooling") {
            constexpr std::size_t width = 4;
            constexpr std::size_t height = 4;
            constexpr std::size_t stride = 2;

            using Grid =
             typename nn::ConvolutionGrid< width, height, nn::Kernel< 2, 2, stride > >::define;
            using PoolingLayer = nn::PoolingLayer< nn::NeuralLayer, nn::Avg, Grid >;

            auto layer = PoolingLayer{};
            WHEN("all 16 inputs are 1.0") {
                for(std::size_t i = 0; i < 16; ++i) {
                    layer.setInput(i, 1.0f);
                }

                THEN("The layer has 4 neurons") {
                    REQUIRE(4 == layer.size());
                }

                WHEN("calculateOutputs is called") {
                    THEN("All outputs should be 1.0 (average of all ones)") {
                        layer.calculateOutputs();
                        for(std::size_t i = 0; i < 4; ++i) {
                            REQUIRE_THAT(1.0f,
                                         Catch::Matchers::WithinRel(layer[i].getOutput(), 0.001f));
                        }
                    }
                }
            }
        }

        GIVEN("A pooling layer with all ones input") {
            constexpr std::size_t width = 4;
            constexpr std::size_t height = 4;
            constexpr std::size_t stride = 2;

            using Grid =
             typename nn::ConvolutionGrid< width, height, nn::Kernel< 2, 2, stride > >::define;
            using PoolingLayer = nn::PoolingLayer< nn::NeuralLayer, nn::Avg, Grid >;

            auto layer = PoolingLayer{};
            WHEN("all 16 inputs are 1.0") {
                for(std::size_t i = 0; i < 16; ++i) {
                    layer.setInput(i, 1.0f);
                }

                WHEN("calculateOutputs is called") {
                    THEN("All outputs should be 1.0 (average of 4 ones)") {
                        layer.calculateOutputs();
                        for(std::size_t i = 0; i < 4; ++i) {
                            REQUIRE_THAT(1.0f,
                                         Catch::Matchers::WithinRel(layer[i].getOutput(), 0.001f));
                        }
                    }
                }
            }
        }
    }

    SCENARIO("Pooling with L2 algorithm", "[layer][pooling][l2]") {
        GIVEN(
         "A pooling layer for an image 4*4, kernel 2*2 stride 2 with L2 "
         "pooling") {
            constexpr std::size_t width = 4;
            constexpr std::size_t height = 4;
            constexpr std::size_t stride = 2;

            using Grid =
             typename nn::ConvolutionGrid< width, height, nn::Kernel< 2, 2, stride > >::define;
            using PoolingLayer = nn::PoolingLayer< nn::NeuralLayer, nn::L2, Grid >;

            auto layer = PoolingLayer{};
            WHEN("inputs are 3, 4 in first frame") {
                layer.setInput(0, 3.0f);
                layer.setInput(1, 4.0f);
                for(std::size_t i = 2; i < 16; ++i) {
                    layer.setInput(i, 0.0f);
                }

                THEN("The layer has 4 neurons") {
                    REQUIRE(4 == layer.size());
                }

                WHEN("calculateOutputs is called") {
                    THEN("First neuron output is sqrt(3^2 + 4^2) = 5") {
                        layer.calculateOutputs();
                        REQUIRE_THAT(5.0f, Catch::Matchers::WithinRel(layer[0].getOutput(), 0.001f));
                    }
                }
            }
        }

        GIVEN("A pooling layer with L2 values") {
            constexpr std::size_t width = 4;
            constexpr std::size_t height = 4;
            constexpr std::size_t stride = 2;

            using Grid =
             typename nn::ConvolutionGrid< width, height, nn::Kernel< 2, 2, stride > >::define;
            using PoolingLayer = nn::PoolingLayer< nn::NeuralLayer, nn::L2, Grid >;

            auto layer = PoolingLayer{};
            WHEN("inputs are 1, 1 in first frame") {
                layer.setInput(0, 1.0f);
                layer.setInput(1, 1.0f);
                for(std::size_t i = 2; i < 16; ++i) {
                    layer.setInput(i, 0.0f);
                }

                WHEN("calculateOutputs is called") {
                    THEN("First neuron output is sqrt(1^2 + 1^2) = sqrt(2)") {
                        layer.calculateOutputs();
                        REQUIRE_THAT(1.41421356f,
                                     Catch::Matchers::WithinRel(layer[0].getOutput(), 0.001f));
                    }
                }
            }
        }
    }

    SCENARIO("Pooling with different kernel sizes", "[layer][pooling][kernel]") {
        GIVEN("A pooling layer with 4x4 kernel") {
            constexpr std::size_t width = 8;
            constexpr std::size_t height = 8;
            constexpr std::size_t stride = 4;

            using Grid =
             typename nn::ConvolutionGrid< width, height, nn::Kernel< 4, 4, stride > >::define;
            using PoolingLayer = nn::PoolingLayer< nn::NeuralLayer, nn::Max, Grid >;

            auto layer = PoolingLayer{};
            WHEN("all inputs from 1 to 64") {
                for(auto i : ranges::views::ints(0, 64)) {
                    layer.setInput(i, static_cast< float >(i + 1));
                }

                THEN(
                 "The layer has 4 neurons (matches ConvolutionLayer grid "
                 "size)") {
                    REQUIRE(4 == layer.size());
                }

                WHEN("calculateOutputs is called") {
                    THEN("Each output is the max of its frame's inputs") {
                        layer.calculateOutputs();
                        REQUIRE_THAT(28.0f, Catch::Matchers::WithinRel(layer[0].getOutput()));
                        REQUIRE_THAT(32.0f, Catch::Matchers::WithinRel(layer[1].getOutput()));
                        REQUIRE_THAT(60.0f, Catch::Matchers::WithinRel(layer[2].getOutput()));
                        REQUIRE_THAT(64.0f, Catch::Matchers::WithinRel(layer[3].getOutput()));
                    }
                }
            }
        }
    }
} // namespace
