#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/Perceptron/Perceptron.h"

#include <range/v3/all.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch.hpp>

#include <vector>
#include <string>

namespace {

    template< typename Memento >
    Memento read(std::istream& strm) {
        Memento memento{};
        cereal::JSONInputArchive archive(strm);
        archive >> memento;
        return memento;
    }

    SCENARIO("Perceptron calculation", "[perceptron][precalculated][forward]") {
        GIVEN(
         "A 3 layer perceptron which has 2 inputs and one output and "
         "precalculated "
         "weights such that perceptron can recognise a XOR gate") {
            using Perceptron =
             nn::Perceptron< float,
                             nn::InputLayer< nn::Neuron, nn::TanhFunction, 2, 1 >,
                             nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 5 >,
                             nn::ComplexNeuralLayer< nn::Neuron< nn::TanhFunction >, nn::Neuron< nn::SigmoidFunction > >,
                             nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 1 > >;

            Perceptron perceptron;
            using Input = typename Perceptron::Input;
            std::ifstream xorJson("SystemTests/etc/xor.json");
            REQUIRE(xorJson.is_open());
            perceptron.setMemento(read< Perceptron::Memento >(xorJson));
            WHEN("Input is [1, 0]") {
                std::array< Input, 2 > input{Input{1}, Input{0}};
                THEN("Output is close to 1") {
                    std::array< float, 2 > output{0};
                    perceptron.calculate(input.begin(), input.end(), output.begin());
                    REQUIRE(output[0] == Approx(1).margin(0.2));
                }
            }
            WHEN("Input is [0, 1]") {
                std::array< Input, 2 > input{Input{0}, Input{1}};
                THEN("Output is close to 1") {
                    std::array< float, 2 > output{0};
                    perceptron.calculate(input.begin(), input.end(), output.begin());
                    REQUIRE(output[0] == Approx(1).margin(0.2));
                }
            }
            WHEN("Input is [1, 1]") {
                std::array< Input, 2 > input{Input{1}, Input{1}};
                THEN("Output is close to 0") {
                    std::array< float, 2 > output{0};
                    perceptron.calculate(input.begin(), input.end(), output.begin());
                    REQUIRE(output[0] == Approx(0).margin(0.2));
                }
            }
            WHEN("Input is [0, 0]") {
                std::array< Input, 2 > input{Input{0}, Input{0}};
                THEN("Output is close to 0") {
                    std::array< float, 2 > output{0};
                    perceptron.calculate(input.begin(), input.end(), output.begin());
                    REQUIRE(output[0] == Approx(0).margin(0.2));
                }
            }
        }
    }

} // namespace
