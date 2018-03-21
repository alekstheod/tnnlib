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

namespace {
    template< typename Neuron >
    bool hasValidInputs(const Neuron& neuron, const std::vector< float >& expected) {
        for(int i = 0; i < expected.size(); i++) {
            if(neuron[i].value != expected[i])
                return false;
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
         nn::ConvolutionLayer< nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 4, 10 >,
                               nn::Connection< nn::Range< 0, 5 >, 0 >,
                               nn::Connection< nn::Range< 2, 7 >, 1 >,
                               nn::Connection< nn::Range< 4, 9 >, 2 >,
                               nn::Connection< nn::Range< 6, 10 >, 3 > >;

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

SCENARIO("Test neuron with sigmoid activation function",
         "[neuron][binary][forward]") {
    GIVEN("A neuron with a binary activation function and 2 inputs") {
        using Neuron = nn::Neuron< nn::SigmoidFunction, float, 2 >;
        WHEN("Input is [0,1,2,3,4,5,6,7,8,9]") {

            THEN("") {
            }
        }
    }
}
