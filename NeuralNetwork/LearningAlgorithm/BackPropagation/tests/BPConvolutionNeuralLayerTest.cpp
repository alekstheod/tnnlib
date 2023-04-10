#include "NeuralNetwork/LearningAlgorithm/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/LearningAlgorithm/BackPropagation/BPConvolutionNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"


#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

namespace {

    constexpr auto width = 5;
    constexpr auto height = 5;
    constexpr auto stride = 2;
    constexpr auto margin = 1;

    using ConvolutionGrid =
     typename nn::ConvolutionGrid< width, height, stride, margin >::define;
    using ConvolutionLayer =
     nn::ConvolutionLayer< nn::NeuralLayer, nn::Neuron, nn::TanhFunction, ConvolutionGrid >;

    using BPConvolutionNeuralLayer = nn::bp::BPNeuralLayer< ConvolutionLayer >;

    SCENARIO("BPConvolutionNeuralLayer weight calculation test",
             "[layer][convolution][backward]") {
        GIVEN("A BPConvolutionNeuralLayer layer 5 neurons and 9 inputs") {
            BPConvolutionNeuralLayer layer;
            WHEN(
             "calculateWeights is called with learning rate 1, deltas set to "
             "0.5, inputs set to 1.0 and weights are 0.5") {
                for(auto& neuron : layer) {
                    neuron.setDelta(0.5f);
                    for(auto i : ranges::views::indices(neuron.size())) {
                        neuron[i].value = 1.f;
                        neuron[i].weight = 0.5f;
                    }
                }

                layer.calculateWeights(1.f);

                // g = in[i]*delta + in[j]*delta ... where in[i..j] interseting
                // intputs w[k] = w[k-1] - g*L;
                const auto& n1 = layer[0];
                const auto& n2 = layer[1];
                const auto& n3 = layer[2];
                const auto& n4 = layer[3];
                THEN("Calculated weights for regular inputs are 0") {

                    REQUIRE_THAT(0.f, Catch::WithinRel(n1[0].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n1[1].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n1[3].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n1[4].weight));

                    REQUIRE_THAT(0.f, Catch::WithinRel(n2[1].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n2[2].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n2[4].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n2[5].weight));

                    REQUIRE_THAT(0.f, Catch::WithinRel(n3[3].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n3[4].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n3[6].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n3[7].weight));

                    REQUIRE_THAT(0.f, Catch::WithinRel(n4[4].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n4[5].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n4[7].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n4[8].weight));
                }

                THEN(
                 "Calculated weights for the once intersecting inputs are "
                 "-0.5") {
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n1[2].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n1[5].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n1[6].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n1[7].weight));

                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n2[0].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n2[3].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n2[7].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n2[8].weight));

                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n3[0].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n3[1].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n3[5].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n3[8].weight));

                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n4[1].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n4[2].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n4[3].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n4[6].weight));
                }

                THEN(
                 "Calculated weights for the 4 times intersecting inputs are "
                 "-1.5") {
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n1[8].weight));
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n2[6].weight));
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n3[2].weight));
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n4[0].weight));
                }
            }
        }
    }
} // namespace
