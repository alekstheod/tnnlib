#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/BPConvolutionNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"


#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

namespace {

    constexpr auto width = 5;
    constexpr auto height = 5;

    using ConvolutionGrid =
     typename nn::ConvolutionGrid< width, height, nn::Kernel< 3, 3, 2 > >::define;

    using ConvolutionLayer =
     nn::ConvolutionLayer< nn::NeuralLayer, nn::Neuron, nn::TanhFunction, ConvolutionGrid >;

    using BPConvolutionNeuralLayer = nn::bp::BPNeuralLayer< ConvolutionLayer >;

    SCENARIO("BPConvolutionNeuralLayer weight calculation test",
             "[layer][convolution][backward]") {
        GIVEN("A BPConvolutionNeuralLayer layer 5 neurons and 9 inputs") {
            auto layer = BPConvolutionNeuralLayer{};
            WHEN(
             "calculateWeights is called with learning rate 1, deltas set to "
             "0.5, inputs set to 1.0 and weights are 0.5") {
                for(auto& neuron : layer) {
                    neuron.setDelta(0.5f);
                    for(auto i : ranges::views::indices(neuron.size())) {
                        neuron[i].weight = 0.5f;
                    }
                }

                for(auto i : ranges::views::ints(0, 25)) {
                    layer.setInput(i, static_cast< float >(1.f));
                }

                for(auto& neuron : layer) {
                    for(auto i : ranges::views::ints(0, 9)) {
                        // neuron[i].weight = 0.5f;
                        std::cout << neuron[i].value << " ";
                    }

                    std::cout << std::endl;
                }

                REQUIRE(9 == layer.size());

                layer.calculateWeights(1.f);

                const auto& n1 = layer[0];
                const auto& n2 = layer[1];
                const auto& n3 = layer[2];

                const auto& n4 = layer[3];
                const auto& n5 = layer[4];
                const auto& n6 = layer[5];

                const auto& n7 = layer[6];
                const auto& n8 = layer[7];
                const auto& n9 = layer[8];
                // g = in[i]*delta + in[j]*delta ... where in[i..j] intersecting
                // intputs w[k] = w[k-1] - g*L;
                THEN("Calculated weight for non intersecting input is 0") {
                    REQUIRE_THAT(0.f, Catch::WithinRel(n1[0].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n1[1].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n1[3].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n1[4].weight));

                    REQUIRE_THAT(0.f, Catch::WithinRel(n2[1].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n2[4].weight));

                    REQUIRE_THAT(0.f, Catch::WithinRel(n4[3].weight));
                    REQUIRE_THAT(0.f, Catch::WithinRel(n4[4].weight));
                }

                THEN("Calculated weight for non connected input is 0.5") {
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n3[1].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n3[2].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n3[4].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n3[5].weight));

                    REQUIRE_THAT(0.5f, Catch::WithinRel(n6[1].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n6[2].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n6[4].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n6[5].weight));

                    REQUIRE_THAT(0.5f, Catch::WithinRel(n8[4].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n8[5].weight));

                    REQUIRE_THAT(0.5f, Catch::WithinRel(n7[3].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n7[4].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n7[5].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n7[6].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n7[7].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n7[8].weight));

                    REQUIRE_THAT(0.5f, Catch::WithinRel(n8[3].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n8[4].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n8[5].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n8[6].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n8[7].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n8[8].weight));

                    REQUIRE_THAT(0.5f, Catch::WithinRel(n9[3].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n9[4].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n9[5].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n9[6].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n9[7].weight));
                    REQUIRE_THAT(0.5f, Catch::WithinRel(n9[8].weight));
                }

                THEN(
                 "Calculated weights for once intersecting inputs are "
                 "-0.5") {
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n1[5].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n1[7].weight));

                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n2[0].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n2[2].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n2[3].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n2[5].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n2[7].weight));

                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n3[0].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n3[3].weight));

                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n4[1].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n4[6].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n4[7].weight));

                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n5[1].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n5[3].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n5[5].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n5[7].weight));

                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n6[3].weight));

                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n7[0].weight));
                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n7[1].weight));

                    REQUIRE_THAT(-0.5f, Catch::WithinRel(n8[1].weight));
                }

                THEN(
                 "Calculated weights for the 4 times intersecting inputs are "
                 "-1.5") {
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n1[8].weight));

                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n2[6].weight));
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n2[8].weight));

                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n3[6].weight));

                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n4[2].weight));
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n4[8].weight));

                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n5[0].weight));
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n5[2].weight));
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n5[6].weight));
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n5[8].weight));

                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n6[0].weight));
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n6[6].weight));

                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n7[2].weight));

                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n8[0].weight));
                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n8[2].weight));

                    REQUIRE_THAT(-1.5f, Catch::WithinRel(n9[0].weight));
                }
            }
        }
    }
} // namespace
