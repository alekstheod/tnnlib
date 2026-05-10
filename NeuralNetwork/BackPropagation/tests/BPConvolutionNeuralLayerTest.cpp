#include "NeuralNetwork/BackPropagation/BPContext.h"
#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/BPConvolutionNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"


#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch_all.hpp>

namespace {

    constexpr auto width = 5;
    constexpr auto height = 5;

    using ConvolutionGrid =
     typename nn::ConvolutionGrid< width, height, nn::Kernel< 3, 3, 2 > >::define;

    using ConvolutionLayer =
     nn::ConvolutionLayer< nn::NeuralLayer, nn::Neuron, nn::TanhFunction, ConvolutionGrid >;

    using BPConvolutionNeuralLayer = nn::bp::BPNeuralLayer< ConvolutionLayer >;
    using BPCtx = nn::bp::BPContext< float, std::tuple< BPConvolutionNeuralLayer > >;
    using Var = typename BPConvolutionNeuralLayer::Var;

    SCENARIO("BPConvolutionNeuralLayer weight calculation test",
             "[layer][convolution][backward]") {
        GIVEN("A BPConvolutionNeuralLayer with 9 neurons and kernel 3x3") {
            auto layer = BPConvolutionNeuralLayer{};
            WHEN(
             "calculateWeights is called with learning rate 1, delta = 0.5, "
             "inputs = 1.0 and weights = 0.5") {
                using Context = std::tuple<std::array<float, 9>>;
                Context ctx;
                BPCtx bpCtx{ctx, {}, {}, {}, {}, {}};
                auto& deltas = std::get< 0 >(bpCtx.deltas);
                auto& weights = std::get< 0 >(bpCtx.weights);
                auto& biases = std::get< 0 >(bpCtx.biases);
                constexpr auto kernelSize = 9;

                for(auto nid : ranges::views::ints(0, 9)) {
                    deltas[nid] = 0.5f;
                    for(auto i : ranges::views::indices(kernelSize)) {
                        weights[nid * kernelSize + i] = 0.5f;
                    }
                    biases[nid] = layer[nid].getBias();
                }

                for(auto i : ranges::views::ints(0, 25)) {
                    layer.setInput(i, static_cast< float >(1.f));
                }

                REQUIRE(9 == layer.size());

                layer.template calculateWeights< BPCtx, 0 >(bpCtx, 1.f);

                THEN("Weights are updated correctly based on input values") {
                    for(auto nid : ranges::views::ints(0, 9)) {
                        for(auto wid : ranges::views::ints(0, 9)) {
                            Var expected = (layer[nid][wid].value > 0.0f) ? 0.0f : 0.5f;
                            REQUIRE_THAT(expected,
                                         Catch::Matchers::WithinRel(
                                          weights[nid * kernelSize + wid]));
                        }
                    }
                }
            }
        }
    }
} // namespace
