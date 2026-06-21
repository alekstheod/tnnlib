#include "NeuralNetwork/Perceptron/PerceptronBuilder.h"
#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"
#include "NeuralNetwork/NeuralLayer/PoolingLayer.h"
#include "NeuralNetwork/Neuron/PoolingNeuron.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include <type_traits>
#include <catch2/catch_all.hpp>

TEST_CASE("PerceptronBuilder basic functionality", "[PerceptronBuilder]") {
    REQUIRE(nn::build< float >().input< 180 >().dense< 30 >().dense< 10 >().size() == 3);
}

TEST_CASE("PerceptronBuilder with_neuron", "[PerceptronBuilder]") {
    REQUIRE(nn::build< float >()
             .input< 180 >()
             .dense< 30 >()
             .dense< 10 >()
             .with_neuron< nn::Neuron< nn::TanhFunction, float > >()
             .size() == 3);
}

TEST_CASE("PerceptronBuilder conv", "[PerceptronBuilder]") {
    using SlidingWindow = nn::SlidingWindow< 8, 8, nn::Kernel< 3, 3, 1 > >;
    REQUIRE(nn::build< float >().input< 64 >().conv< SlidingWindow >().size() == 2);
}

TEST_CASE("PerceptronBuilder conv default", "[PerceptronBuilder]") {
    REQUIRE(nn::build< float >().input< 64 >().conv().build().size() == 2);
}

TEST_CASE("PerceptronBuilder conv with_kernel", "[PerceptronBuilder]") {
    REQUIRE(nn::build< float >()
             .input< 64 >()
             .conv()
             .template with_kernel< 2, 2, 1 >()
             .build()
             .size() == 2);
}

TEST_CASE("PerceptronBuilder conv with_grid", "[PerceptronBuilder]") {
    REQUIRE(nn::build< float >()
             .input< 64 >()
             .conv()
             .template with_grid< 4, 4 >()
             .build()
             .size() == 2);
}

TEST_CASE("PerceptronBuilder conv with_kernel and with_grid",
          "[PerceptronBuilder]") {
    REQUIRE(nn::build< float >()
             .input< 64 >()
             .conv()
             .with_kernel< 3, 3, 1 >()
             .with_grid< 8, 8 >()
             .build()
             .dense< 20 >()
             .size() == 3);
}

TEST_CASE("PerceptronBuilder pool max", "[PerceptronBuilder]") {
    using SlidingWindow = nn::SlidingWindow< 8, 8, nn::Kernel< 2, 2, 2 > >;
    REQUIRE(
     nn::build< float >().input< 64 >().template pool< nn::Max, SlidingWindow >().size() == 2);
}

TEST_CASE("PerceptronBuilder pool avg", "[PerceptronBuilder]") {
    using SlidingWindow = nn::SlidingWindow< 8, 8, nn::Kernel< 2, 2, 2 > >;
    REQUIRE(
     nn::build< float >().input< 64 >().template pool< nn::Avg, SlidingWindow >().size() == 2);
}

TEST_CASE("PerceptronBuilder pool after conv", "[PerceptronBuilder]") {
    using SlidingWindow = nn::SlidingWindow< 8, 8, nn::Kernel< 3, 3, 1 > >;
    using SlidingWindow2 = nn::SlidingWindow< 6, 6, nn::Kernel< 2, 2, 2 > >;
    REQUIRE(nn::build< float >()
             .input< 64 >()
             .conv< SlidingWindow >()
             .template pool< nn::Max, SlidingWindow2 >()
             .dense< 10 >()
             .size() == 4);
}

TEST_CASE("PerceptronBuilder async", "[PerceptronBuilder]") {
    REQUIRE(nn::build< float >()
             .input< 180 >()
             .template async< 30 >()
             .template async< 10 >()
             .size() == 3);
}
