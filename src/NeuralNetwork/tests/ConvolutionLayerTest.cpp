#include <NeuralNetwork/NeuralLayer/ConvolutionLayer.h>
#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/Neuron/Neuron.h>

#include <range/v3/all.hpp>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch.hpp>

namespace {
    template< typename Neuron >
    bool hasValidInputs(const Neuron& neuron, const std::vector< float >& expected) {
        for(int i = 0; i < expected.size(); i++) {
            if(neuron[i].value != expected[i])
                return false;
        }

        return true;
    }
} // namespace

SCENARIO("Perceptron calculation with convolution layer",
         "[perceptron][convolution][forward]") {
    GIVEN(
     "A convolution layer with  the given set of connections:\n"
     "[0-5]->0,\n"
     "[2-7]->2,\n"
     "[4-9]->3,\n"
     "[6-10]->4") {
        using ConvolutionLayer =
         nn::ConvolutionLayer< nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 4, 10 >,
                               nn::Connection< nn::InputRange< 0, 5 >, 0 >,
                               nn::Connection< nn::InputRange< 2, 7 >, 1 >,
                               nn::Connection< nn::InputRange< 4, 9 >, 2 >,
                               nn::Connection< nn::InputRange< 6, 10 >, 3 > >;

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