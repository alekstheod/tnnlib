#include "NeuralNetwork/NeuralLayer/Thread/AsyncNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"

#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch_all.hpp>


namespace {
    SCENARIO("AsyncNeuralLayer compared to regular NeuralLayer",
             "[layer][thread][forward]") {
        GIVEN(
         "A AsyncNeuralLayer layer with 2 neurons and 2 inputs as well as a "
         "regular layer with the same topology") {
            nn::NeuralLayer< nn::Neuron, nn::TanhFunction, 2, 2 > regularLayer;
            nn::AsyncNeuralLayer< nn::Neuron, nn::TanhFunction, 2, 2 > asyncLayer;
            const auto memento = regularLayer.getMemento();
            asyncLayer.setMemento(memento);

            regularLayer.setInput(0, 0.1f);
            regularLayer.setInput(1, 0.2f);

            asyncLayer.setInput(0, 0.1f);
            asyncLayer.setInput(1, 0.2f);

            WHEN("The weights and the inputs of both layers are identical") {
                THEN("The output is of both layers identical") {
                    regularLayer.calculateOutputs();
                    asyncLayer.calculateOutputs();
                    const auto expected_output = regularLayer.getOutput(0);
                    const auto actual_output = asyncLayer.getOutput(0);
                    REQUIRE_THAT(expected_output, Catch::WithinRel(actual_output));
                }
            }
        }
    }
} // namespace
