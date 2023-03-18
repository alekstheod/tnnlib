#include <NeuralNetwork/NeuralLayer/AsyncNeuralLayer.h>
#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/ActivationFunction/TanhFunction.h>
#include <NeuralNetwork/Neuron/Neuron.h>

#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>


namespace {
    SCENARIO("AsyncNeuralLayer compared to regular NeuralLayer",
             "[layer][thread][forward]") {
        GIVEN(
         "A AsyncNeuralLayer layer with 2 neurons and 2 inputs as well as a "
         "regular layer with the same topology") {
            nn::NeuralLayer< nn::Neuron, nn::TanhFunction, 2, 2 > regularLayer;
            const auto memento = regularLayer.getMemento();
            nn::AsyncNeuralLayer< nn::Neuron, nn::TanhFunction, 2, 2 > asyncLayer;
            asyncLayer.setMemento(memento);
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
