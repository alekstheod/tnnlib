#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/TanhFunction.h>
#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/Perceptron/Perceptron.h>

#include <etc/xorXml.h>

#include <cereal/archives/xml.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <array>
#include <sstream>

namespace {
    template< typename Memento >
    Memento read(const std::string& str) {
        Memento memento{};
        std::stringstream strm(str);
        cereal::XMLInputArchive archive(strm);
        archive >> memento;
        return memento;
    }
} // namespace

SCENARIO("Perceptron calculation", "[perceptron][precalculated][forward]") {
    GIVEN(
     "A 2 layer perceptron which has 2 inputs and one output and precalculated "
     "weights such that perceptron can recognise a XOR gate") {
        using Perceptron =
         nn::Perceptron< float,
                         nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 2 >,
                         nn::NeuralLayer< nn::Neuron, nn::TanhFunction, 3 >,
                         nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 1 > >;

        Perceptron perceptron;
        perceptron.setMemento(read< Perceptron::Memento >(xorXml));
        WHEN("Input is [1, 0]") {
            std::array< float, 2 > input{1, 0};
            THEN("Output is close to 1") {
                std::array< float, 2 > output{0};
                perceptron.calculate(input.begin(), input.end(), output.begin());
                REQUIRE(output[0] == Approx(1).margin(0.1));
            }
        }
        WHEN("Input is [0, 1]") {
            std::array< float, 2 > input{0, 1};
            THEN("Output is close to 1") {
                std::array< float, 2 > output{0};
                perceptron.calculate(input.begin(), input.end(), output.begin());
                REQUIRE(output[0] == Approx(1).margin(0.1));
            }
        }
        WHEN("Input is [1, 1]") {
            std::array< float, 2 > input{1, 1};
            THEN("Output is close to 0") {
                std::array< float, 2 > output{0};
                perceptron.calculate(input.begin(), input.end(), output.begin());
                REQUIRE(output[0] == Approx(0).margin(0.1));
            }
        }
        WHEN("Input is [0, 0]") {
            std::array< float, 2 > input{0, 0};
            THEN("Output is close to 0") {
                std::array< float, 2 > output{0};
                perceptron.calculate(input.begin(), input.end(), output.begin());
                REQUIRE(output[0] == Approx(0).margin(0.1));
            }
        }
    }
}