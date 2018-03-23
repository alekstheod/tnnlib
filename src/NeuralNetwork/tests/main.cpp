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

SCENARIO("Convolution layer set inputs", "[layer][convolution][forward]") {
    GIVEN(
     "A convolution layer with a set of connections:\n"
     "[0-5]->0,\n"
     "[2-7]->2,\n"
     "[4-9]->3,\n"
     "[6-10]->4") {
        using ConvolutionLayer =
         nn::ConvolutionLayer< nn::NeuralLayer,
                               nn::Neuron,
                               nn::SigmoidFunction,
                               10,
                               std::tuple< nn::Connection< nn::Range< 0, 5 >, 0 >,
                                           nn::Connection< nn::Range< 2, 7 >, 1 >,
                                           nn::Connection< nn::Range< 4, 9 >, 2 >,
                                           nn::Connection< nn::Range< 6, 10 >, 3 > > >;

        WHEN("Input is [0,1,2,3,4,5,6,7,8,9]") {
            ConvolutionLayer layer;
            for(const auto input : ranges::v3::view::ints(0, 10)) {
                layer.setInput(input, input);
            }
            THEN(
             "neuron 0 has inputs = [0,1,2,3,4,0,0,0,0,0]\n"
             "neuron 1 has inputs = [0,0,2,3,4,5,6,0,0,0]\n"
             "neuron 2 has inputs = [0,0,0,0,4,5,6,7,8,0]\n"
             "neuron 3 has inputs = [0,0,0,0,0,0,6,7,8,9]\n") {
                std::vector< std::vector< float > > inputs = {
                 {0, 1, 2, 3, 4, 0, 0, 0, 0, 0},
                 {0, 0, 2, 3, 4, 5, 6, 0, 0, 0},
                 {0, 0, 0, 0, 4, 5, 6, 7, 8, 0},
                 {0, 0, 0, 0, 0, 0, 6, 7, 8, 9}};

                for(const auto id : ranges::v3::view::ints(0, 4)) {
                    REQUIRE(hasValidInputs(layer[id], inputs[id]));
                }
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
                // clang-format off
                std::vector< std::vector< float > > inputs = {
                 {1,  2,  3,  0,  0,  
                  6,  7,  8,  0,  0, 
                  11, 12, 13, 0,  0, 
                  0,  0,  0,  0,  0, 
                  0,  0,  0,  0,  0},
                 {0,  0,  3,  4,  5,  
                  0,  0,  8,  9,  10, 
                  0,  0,  13, 14, 15, 
                  0,  0,  0,  0,  0, 
                  0,  0,  0,  0,  0},
                 {0,  0,  0,  0,  0,  
                  0,  0,  0,  0,  0, 
                  11, 12, 13, 0,  0, 
                  16, 17, 18, 0,  0, 
                  21, 22, 23, 0,  0},
                 {0,  0,  0,  0,  0,  
                  0,  0,  0,  0,  0, 
                  0,  0,  13, 14, 15, 
                  0,  0,  18, 19, 20, 
                  0,  0,  23, 24, 25}};
                // clang-format on
                for(const auto id : ranges::v3::view::ints(0, 4)) {
                    REQUIRE(hasValidInputs(layer[0], inputs[0]));
                }
            }
        }
    }
}
