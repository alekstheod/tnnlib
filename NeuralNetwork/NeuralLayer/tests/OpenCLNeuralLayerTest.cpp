#include "NeuralNetwork/NeuralLayer/OpenCL/OpenCLNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"

#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

namespace {
    SCENARIO("OpenCLNeuralLayer compared to regular NeuralLayer",
             "[layer][opencl][forward]") {
        GIVEN(
         "A OpenCLNeuralLayer layer with 2 neurons and 2 inputs as well as a "
         "regular layer with the same topology") {
            nn::NeuralLayer< nn::Neuron, nn::TanhFunction, 2, 2 > regularLayer;
            nn::OpenCLNeuralLayer< nn::Neuron, nn::TanhFunction, 2, 2 > openClLayer;
            const auto memento = regularLayer.getMemento();
            openClLayer.setMemento(memento);

            regularLayer[0][0].value = 0.1f;
            regularLayer[0][1].value = 0.2f;
            regularLayer[1][0].value = 0.3f;
            regularLayer[1][1].value = 0.4f;

            openClLayer[0][0].value = 0.1f;
            openClLayer[0][1].value = 0.2f;
            openClLayer[1][0].value = 0.3f;
            openClLayer[1][1].value = 0.4f;

            WHEN("The weights and the inputs of both layers are identical") {
                THEN("The output is of both layers identical") {
                    regularLayer.calculateOutputs();
                    openClLayer.calculateOutputs();
                    const auto expected_output = regularLayer.getOutput(0);
                    const auto actual_output = openClLayer.getOutput(0);
                    std::cout << "Outputs " << expected_output << " "
                              << actual_output << std::endl;
                    REQUIRE_THAT(expected_output, Catch::WithinRel(actual_output));
                }
            }
        }
    }
} // namespace
