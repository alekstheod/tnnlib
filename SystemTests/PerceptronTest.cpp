#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/Perceptron/Perceptron.h"
#include "NeuralNetwork/Serialization/Cereal.h"

#include <range/v3/all.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <fstream>

namespace {

    template< typename Memento >
    Memento read(std::istream& strm) {
        Memento memento{};
        cereal::JSONInputArchive archive(strm);
        archive >> cereal::make_nvp("perceptron", memento);
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
                    REQUIRE(output[0] == Catch::Approx(1).margin(0.1));
                }
            }
            WHEN("Input is [0, 1]") {
                std::array< Input, 2 > input{Input{0}, Input{1}};
                THEN("Output is close to 1") {
                    std::array< float, 2 > output{0};
                    perceptron.calculate(input.begin(), input.end(), output.begin());
                    REQUIRE(output[0] == Catch::Approx(1).margin(0.1));
                }
            }
            WHEN("Input is [1, 1]") {
                std::array< Input, 2 > input{Input{1}, Input{1}};
                THEN("Output is close to 0") {
                    std::array< float, 2 > output{0};
                    perceptron.calculate(input.begin(), input.end(), output.begin());
                    REQUIRE(output[0] == Catch::Approx(0).margin(0.1));
                }
            }
            WHEN("Input is [0, 0]") {
                std::array< Input, 2 > input{Input{0}, Input{0}};
                THEN("Output is close to 0") {
                    std::array< float, 2 > output{0};
                    perceptron.calculate(input.begin(), input.end(), output.begin());
                    REQUIRE(output[0] == Catch::Approx(0).margin(0.1));
                }
            }
        }
    }

    SCENARIO("Perceptron input distribution",
             "[perceptron][input][regression]") {
        GIVEN("A perceptron with 3 neurons, 2 features each") {
            using TestPerceptron =
             nn::Perceptron< float,
                             nn::InputLayer< nn::Neuron, nn::SigmoidFunction, 3, 2 >,
                             nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 1 > >;

            TestPerceptron perceptron;
            using Input = typename TestPerceptron::Input;

            auto memento = perceptron.getMemento();
            auto& inputLayerMemento = std::get< 0 >(memento.layers);
            for(auto& neuron : inputLayerMemento.neurons) {
                neuron.bias = 0.0f;
                neuron.weights[0] = 1.0f;
                neuron.weights[1] = 0.0f;
            }
            auto& hiddenLayerMemento = std::get< 1 >(memento.layers);
            hiddenLayerMemento.neurons[0].bias = 0.0f;
            for(int i = 0; i < 3; ++i) {
                hiddenLayerMemento.neurons[0].weights[i] = 1.0f;
            }
            perceptron.setMemento(memento);

            WHEN(
             "Sum of distinct input values through sigmoid should be correct") {
                std::array< Input, 3 > input;
                input[0] = Input{{2.0f, 0.0f}};
                input[1] = Input{{0.5f, 0.0f}};
                input[2] = Input{{0.2f, 0.0f}};

                std::array< float, 1 > output{0.f};
                perceptron.calculate(input.begin(), input.end(), output.begin());

                float n0 = 1.0f / (1.0f + std::exp(-2.0f));
                float n1 = 1.0f / (1.0f + std::exp(-0.5f));
                float n2 = 1.0f / (1.0f + std::exp(-0.2f));
                float expectedSum = n0 + n1 + n2;
                float expected = 1.0f / (1.0f + std::exp(-expectedSum));

                THEN("Output should equal sigmoid(sum of sigmoid inputs)") {
                    REQUIRE(output[0] == Catch::Approx(expected).margin(0.01));
                }
            }

            WHEN("Each input contributes to its corresponding neuron") {
                std::array< Input, 3 > input;
                input[0] = Input{{5.0f, 5.0f}};
                input[1] = Input{{1.0f, 1.0f}};
                input[2] = Input{{0.1f, 0.1f}};

                std::array< float, 1 > output{0.f};
                perceptron.calculate(input.begin(), input.end(), output.begin());

                THEN("Output should sum all input values through sigmoid") {
                    float n0 = 1.0f / (1.0f + std::exp(-10.0f));
                    float n1 = 1.0f / (1.0f + std::exp(-2.0f));
                    float n2 = 1.0f / (1.0f + std::exp(-0.2f));
                    float sum = n0 + n1 + n2;
                    float expected = 1.0f / (1.0f + std::exp(-sum));
                    REQUIRE(output[0] == Catch::Approx(expected).margin(0.01));
                }
            }
        }
    }

} // namespace
