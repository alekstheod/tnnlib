#include "NeuralNetwork/BackPropagation/BPAsyncNeuralLayer.h"
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

    SCENARIO("BPAsyncNeuralLayer compared to regular BPNeuralLayer",
             "[layer][thread][backward]") {
        GIVEN(
         "A AsyncNeuralLayer layer with 2 neurons and 2 inputs as well as a "
         "regular layer with the same topology") {
            nn::bp::BPNeuralLayer< BasicLayer > regularLayer;
            regularLayer.setInput(0, 0.5f);
            regularLayer.setInput(1, 0.3f);

            nn::bp::BPNeuralLayer< nn::detail::AsyncNeuralLayer< BasicLayer > > asyncLayer;
            asyncLayer.setMemento(regularLayer.getMemento());
            asyncLayer.setInput(0, 0.5f);
            asyncLayer.setInput(1, 0.3f);

            Prototype prototype{{0.1f, 0.2f}, {1.f, 1.f}};
            const auto momentum = [](auto, auto newDelta) { return newDelta; };
            regularLayer.calculateOutputs();
            asyncLayer.calculateOutputs();
            WHEN("The weights identical") {
                THEN("The deltas of both layers is identical") {
                    regularLayer.calculateDeltas(prototype, momentum);
                    asyncLayer.calculateDeltas(prototype, momentum);
                    utils::for_< 2 >([&](auto i) {
                        const auto expected_delta = regularLayer.getDelta(i.value);
                        const auto actual_delta = asyncLayer.getDelta(i.value);
                        REQUIRE_THAT(expected_delta, Catch::Matchers::WithinRel(actual_delta));
                    });
                }
            }
            WHEN("Deltas of both layers are identical") {
                regularLayer.calculateDeltas(prototype, momentum);
                asyncLayer.calculateDeltas(prototype, momentum);
                THEN("Calculated weights are identical") {
                    regularLayer.calculateWeights(0.001);
                    asyncLayer.calculateWeights(0.001);
                    utils::for_< 2 >([&](auto i) {
                        utils::for_< 2 >([&](auto j) {
                            const auto expectedWeight =
                             regularLayer[i.value][j.value].weight;
                            const auto actualWeight =
                             asyncLayer[i.value][j.value].weight;
                            const auto expectedBias = regularLayer[i.value].getBias();
                            const auto actualBias = asyncLayer[i.value].getBias();

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
