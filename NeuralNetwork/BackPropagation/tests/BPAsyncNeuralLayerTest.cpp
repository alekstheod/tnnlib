#include "NeuralNetwork/BackPropagation/BPAsyncNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/BPContext.h"
#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/Thread/AsyncNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"


#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch_all.hpp>

namespace {

    using Prototype =
     typename std::tuple< std::array< float, 2 >, std::array< float, 2 > >;
    using BasicLayer = nn::NeuralLayer< nn::Neuron, nn::TanhFunction, 2, 2 >;
    using BPBasicLayer = nn::bp::BPNeuralLayer< BasicLayer >;
    using BPCtx = nn::bp::BPContext< float, std::tuple< BPBasicLayer > >;

    SCENARIO("BPAsyncNeuralLayer compared to regular BPNeuralLayer",
             "[layer][thread][backward]") {
        GIVEN(
         "A AsyncNeuralLayer layer with 2 neurons and 2 inputs as well as a "
         "regular layer with the same topology") {
            BPBasicLayer regularLayer;
            regularLayer[0][0].value = 0.5f;
            regularLayer[0][1].value = 0.3f;
            regularLayer[1][0].value = 0.5f;
            regularLayer[1][1].value = 0.3f;

            nn::bp::BPNeuralLayer< nn::detail::AsyncNeuralLayer< BasicLayer > > asyncLayer;
            asyncLayer.setMemento(regularLayer.getMemento());
            asyncLayer[0][0].value = 0.5f;
            asyncLayer[0][1].value = 0.3f;
            asyncLayer[1][0].value = 0.5f;
            asyncLayer[1][1].value = 0.3f;

            Prototype prototype{{0.1f, 0.2f}, {1.f, 1.f}};
            const auto momentum = [](auto, auto newDelta) { return newDelta; };
            using Context = std::tuple<std::array<float, 2>>;
            Context regCtx, asyncCtx;
            regularLayer.calculateOutputs<Context, 0>(regCtx);
            asyncLayer.calculateOutputs<Context, 0>(asyncCtx);
            WHEN("The weights identical") {
                THEN("The deltas of both layers is identical") {
                    BPCtx regBpCtx{regCtx, {}, {}, {}, {}, {}};
                    BPCtx asyncBpCtx{asyncCtx, {}, {}, {}, {}, {}};
                    auto& regW = std::get<0>(regBpCtx.weights);
                    auto& regB = std::get<0>(regBpCtx.biases);
                    auto& asyncW = std::get<0>(asyncBpCtx.weights);
                    auto& asyncB = std::get<0>(asyncBpCtx.biases);
                    for (auto i : {0, 1}) {
                        regB[i] = regularLayer[i].getBias();
                        asyncB[i] = asyncLayer[i].getBias();
                        for (auto j : {0, 1}) {
                            regW[i * 2 + j] = regularLayer[i][j].weight;
                            asyncW[i * 2 + j] = asyncLayer[i][j].weight;
                        }
                    }
                    regularLayer.template calculateDeltas< BPCtx, 0 >(regBpCtx, prototype, momentum);
                    asyncLayer.template calculateDeltas< BPCtx, 0 >(asyncBpCtx, prototype, momentum);
                    utils::for_< 2 >([&](auto i) {
                        const auto expected_delta = std::get< 0 >(regBpCtx.deltas)[i.value];
                        const auto actual_delta = std::get< 0 >(asyncBpCtx.deltas)[i.value];
                        REQUIRE_THAT(expected_delta, Catch::Matchers::WithinRel(actual_delta));
                    });
                }
            }
            WHEN("Deltas of both layers are identical") {
                BPCtx regBpCtx{regCtx, {}, {}, {}, {}, {}};
                BPCtx asyncBpCtx{asyncCtx, {}, {}, {}, {}, {}};
                auto& regW = std::get<0>(regBpCtx.weights);
                auto& regB = std::get<0>(regBpCtx.biases);
                auto& asyncW = std::get<0>(asyncBpCtx.weights);
                auto& asyncB = std::get<0>(asyncBpCtx.biases);
                for (auto i : {0, 1}) {
                    regB[i] = regularLayer[i].getBias();
                    asyncB[i] = asyncLayer[i].getBias();
                    for (auto j : {0, 1}) {
                        regW[i * 2 + j] = regularLayer[i][j].weight;
                        asyncW[i * 2 + j] = asyncLayer[i][j].weight;
                    }
                }
                regularLayer.template calculateDeltas< BPCtx, 0 >(regBpCtx, prototype, momentum);
                asyncLayer.template calculateDeltas< BPCtx, 0 >(asyncBpCtx, prototype, momentum);
                THEN("Calculated weights are identical") {
                    regularLayer.template calculateWeights< BPCtx, 0 >(regBpCtx, 0.001f);
                    asyncLayer.template calculateWeights< BPCtx, 0 >(asyncBpCtx, 0.001f);
                    utils::for_< 2 >([&](auto i) {
                        utils::for_< 2 >([&](auto j) {
                            const auto expectedWeight =
                             std::get< 0 >(regBpCtx.weights)[i.value * 2 + j.value];
                            const auto actualWeight =
                             std::get< 0 >(asyncBpCtx.weights)[i.value * 2 + j.value];
                            const auto expectedBias = std::get< 0 >(regBpCtx.biases)[i.value];
                            const auto actualBias = std::get< 0 >(asyncBpCtx.biases)[i.value];

                            REQUIRE_THAT(expectedWeight,
                                         Catch::Matchers::WithinRel(actualWeight));
                            REQUIRE_THAT(expectedBias, Catch::Matchers::WithinRel(actualBias));
                        });
                    });
                }
            }
        }
    }
} // namespace
