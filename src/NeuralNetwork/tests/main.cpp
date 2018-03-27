#include <NeuralNetwork/NeuralLayer/ConvolutionLayer.h>
#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/TanhFunction.h>
#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/Perceptron/Perceptron.h>

#include <range/v3/all.hpp>

#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch.hpp>

#include <etc/xorXml.h>

#include <cereal/archives/xml.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

#include <vector>
#include <iostream>

namespace {
    template< typename Neuron >
    bool hasValidInputs(const Neuron& neuron, const std::vector< float >& expected) {
        for(int i = 0; i < expected.size(); i++) {
            if(neuron[i].value != expected[i] || neuron[i].weight != 0) {
                return false;
            }
        }

        return true;
    }

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
     "A 3 layer perceptron which has 2 inputs and one output and "
     "precalculated "
     "weights such that perceptron can recognise a XOR gate") {
        using Perceptron =
         nn::Perceptron< float,
                         nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 2, 2 >,
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

SCENARIO("Convolution grid set inputs", "[layer][convolution][grid][forward]") {
    GIVEN("A convolution layer for an image 5*5, stride = 2 and margin = 1") {
        constexpr std::size_t width = 5;
        constexpr std::size_t height = 5;
        constexpr std::size_t inputsNumber = width * height;
        constexpr std::size_t margin = 1;
        constexpr std::size_t stride = 2;
        using ConvolutionGrid =
         typename nn::ConvolutionGrid< width, height, stride, margin >::define;

        using ConvolutionLayer =
         nn::ConvolutionLayer< nn::NeuralLayer, nn::Neuron, nn::SigmoidFunction, 25, ConvolutionGrid >;
        auto layer = ConvolutionLayer{};
        WHEN(
         "input is a grid filled with the increasing sequence of integers") {
            for(auto i : ranges::v3::view::ints(0, 25)) {
                layer.setInput(i, i + 1);
            }

            THEN("The expected inputs can be described as a following") {
                std::vector< std::vector< float > > inputs = {
                 {1, 2, 3, 6, 7, 8, 11, 12, 13},
                 {3, 4, 5, 8, 9, 10, 13, 14, 15},
                 {11, 12, 13, 16, 17, 18, 21, 22, 23},
                 {13, 14, 15, 18, 19, 20, 23, 24, 25}};
                for(const auto id : ranges::v3::view::ints(0, 4)) {
                    REQUIRE(hasValidInputs(layer[id], inputs[id]));
                }
            }
        }
    }
}

SCENARIO("Neuron output calculation", "[neuron][forward]") {
    GIVEN("Neuron with 3 inputs and sigmoid activation function") {
        nn::Neuron< nn::SigmoidFunction, float, 3 > neuron;
        std::vector< float > dotProducts;
        WHEN("Weights are set to 1 and bias is equal to 1") {
            neuron.setBias(1);
            for(auto i : ranges::v3::view::ints(0, 3)) {
                neuron.setWeight(i, 1);
                neuron.setInput(i, 1);
            }

            THEN("The output of the neuron is 1/(1+exp(-4)) -> 0.982") {
                const auto output =
                 neuron.calculateOutput(dotProducts.begin(), dotProducts.end());
                REQUIRE(output == Approx(0.982f).margin(0.001));
            }
        }
        WHEN("Weights are set to 0 and bias is equal to 0") {
            neuron.setBias(0);
            for(auto i : ranges::v3::view::ints(0, 3)) {
                neuron.setWeight(i, 0);
                neuron.setInput(i, 1);
            }
            THEN("The output of the neuron is 1/1+exp(0) -> 0.5") {
                const auto output =
                 neuron.calculateOutput(dotProducts.begin(), dotProducts.end());
                REQUIRE(output == Approx(0.5f).margin(0.001));
            }
        }
        WHEN(
         "Weights are set to 1 bias is equal to 0 and inputs are set to 0") {
            neuron.setBias(0);
            for(auto i : ranges::v3::view::ints(0, 3)) {
                neuron.setWeight(i, 1);
                neuron.setInput(i, 0);
            }
            THEN("The output of the neuron is 1/1+exp(0) -> 0.5") {
                const auto output =
                 neuron.calculateOutput(dotProducts.begin(), dotProducts.end());
                REQUIRE(output == Approx(0.5f).margin(0.001));
            }
        }
    }

    GIVEN("Neuron with 3 inputs and tahn activation function") {
        nn::Neuron< nn::TanhFunction, float, 3 > neuron;
        WHEN("Weights are set to 1 and bias is equal to 1") {
            THEN("The output of neuron is") {
            }
        }
    }
}
