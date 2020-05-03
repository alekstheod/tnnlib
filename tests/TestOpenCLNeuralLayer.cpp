#include <NeuralNetwork/NeuralLayer/OpenCLNeuralLayer.h>
#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/Neuron/ActivationFunction/TanhFunction.h>
#include <NeuralNetwork/Neuron/Neuron.h>

#include <range/v3/all.hpp>

#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include "catch.hpp"

namespace {
    SCENARIO("OpenCLNeuralLayer compared to regular NeuralLayer",
             "[layer][opencl][forward]") {
        GIVEN(
         "A OpenCLNeuralLayer layer with 2 neurons and 2 inputs as well as a "
         "regular layer with the same topology") {
            nn::NeuralLayer< nn::Neuron, nn::TanhFunction, 2, 2 > regularLayer;
            const auto memento = regularLayer.getMemento();
            nn::OpenCLNeuralLayer< nn::Neuron, nn::TanhFunction, 2, 2 > openClLayer;
            openClLayer.setMemento(memento);
            WHEN("The weights and the inputs of both layers are identical") {
                THEN("The output is of both layers identical") {
                    regularLayer.calculateOutputs();
                    openClLayer.calculateOutputs();
                    const auto expected_output = regularLayer.getOutput(0);
                    const auto actual_output = openClLayer.getOutput(0);
                    REQUIRE(expected_output == actual_output);
                }
            }
        }
    }
} // namespace
