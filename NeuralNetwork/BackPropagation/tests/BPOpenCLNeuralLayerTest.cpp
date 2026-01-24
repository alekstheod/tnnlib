#include "NeuralNetwork//BackPropagation/OpenCL/BPOpenCLNeuralLayer.h"
#include "NeuralNetwork//BackPropagation/BPNeuralLayer.h"
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

            nn::bp::BPNeuralLayer< nn::detail::OpenCLNeuralLayer< BasicLayer > > openclLayer;
            openclLayer.setMemento(regularLayer.getMemento());
            openclLayer.setInput(0, 0.5f);
            openclLayer.setInput(1, 0.3f);

            Prototype prototype{{0.1f, 0.2f}, {1.f, 1.f}};
            const auto momentum = [](auto, auto newDelta) { return newDelta; };
            regularLayer.calculateOutputs();
            openclLayer.calculateOutputs();
            WHEN("The weights identical") {
                THEN("The deltas of both layers are identical") {
                    regularLayer.calculateDeltas(prototype, momentum);
                    openclLayer.calculateDeltas(prototype, momentum);
                    utils::for_< 2 >([&](auto i) {
                        const auto expected_delta = regularLayer.getDelta(i.value);
                        const auto actual_delta = openclLayer.getDelta(i.value);
                        REQUIRE_THAT(expected_delta, Catch::WithinRel(actual_delta));
                    });
                }
            }
            WHEN("Deltas of both layers are identical") {
                regularLayer.calculateDeltas(prototype, momentum);
                openclLayer.calculateDeltas(prototype, momentum);
                THEN("Calculated weights are identical") {
                    regularLayer.calculateWeights(0.001);
                    openclLayer.calculateWeights(0.001);
                    utils::for_< 2 >([&](auto i) {
                        utils::for_< 2 >([&](auto j) {
                            const auto expectedWeight =
                             regularLayer[i.value][j.value].weight;
                            const auto actualWeight =
                             openclLayer.getWeight(i.value, j.value);
                            const auto expectedBias = regularLayer[i.value].getBias();
                            const auto actualBias = openclLayer[i.value].getBias();

                            REQUIRE_THAT(expectedWeight, Catch::WithinRel(actualWeight));
                            REQUIRE_THAT(expectedBias, Catch::WithinRel(actualBias));
                        });
                    });
                }
            }
        }
    }
} // namespace
