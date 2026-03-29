#include "NeuralNetwork/Perceptron/PerceptronBuilder.h"
#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"
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
