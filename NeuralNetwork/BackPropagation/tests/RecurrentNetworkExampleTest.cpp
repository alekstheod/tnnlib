#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/BackPropagation/BepAlgorithm.h"
#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/Perceptron/Perceptron.h"

#include <array>
#include <cmath>
#include <cstddef>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch_all.hpp>

namespace {

    using namespace nn;

    using InputLayer = nn::InputLayer< nn::Neuron, nn::SigmoidFunction, 2, 1 >;
    using HiddenLayer = nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 2 >;
    using OutputLayer = nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 1 >;
    using ElmanNetwork = nn::Perceptron< float, InputLayer, HiddenLayer, OutputLayer >;
    using Algorithm = nn::bp::BepAlgorithm< ElmanNetwork >;
    using Input = ElmanNetwork::Input;

    float sigmoid(float value) {
        return 1.f / (1.f + std::exp(-value));
    }

    bp::ConnectionData elmanConnections() {
        bp::ConnectionData connections;
        connections.set< 0, 1 >(1, connect(1, 0));
        connections.set< 1, 0 >(1, connect(1, 0));
        connections.set< 1, 1 >(1, connect(1, 1));
        return connections;
    }

    struct ReferenceElman {
        float hidden = 0.f;
        float context = 0.f;

        float step(float value) {
            const float input = sigmoid(value);
            const float nextHidden = sigmoid(input + 0.25f * hidden);
            const float nextContext = sigmoid(hidden + 0.25f * context);
            hidden = nextHidden;
            context = nextContext;
            return sigmoid(0.5f * hidden + context);
        }
    };

    void configureAnalyticWeights(Algorithm& algorithm) {
        auto& ctx = algorithm.context();
        std::get< 1 >(ctx.weights) = {1.f, 0.25f, 1.f, 0.25f};
        std::get< 1 >(ctx.biases) = {0.f, 0.f};
        std::get< 2 >(ctx.weights) = {0.5f, 1.f};
        std::get< 2 >(ctx.biases) = {0.f};
    }

    void configureReadoutWeights(Algorithm& algorithm) {
        auto& ctx = algorithm.context();
        std::get< 1 >(ctx.weights) = {10.f, 0.f, 10.f, 0.1f};
        std::get< 1 >(ctx.biases) = {-5.f, -5.f};
        std::get< 2 >(ctx.weights) = {0.f, 0.1f};
        std::get< 2 >(ctx.biases) = {0.f};
    }

    Algorithm::Prototype makeDelayedCopyPrototype(bool previousHigh, bool currentHigh) {
        constexpr float high = 6.f;
        constexpr float low = -6.f;
        const float value = currentHigh ? high : low;
        const float target = previousHigh ? 1.f : 0.f;
        return Algorithm::Prototype{{Input{value}, Input{value}}, {target}};
    }

    SCENARIO("Elman network: closed-form recurrent evaluation",
             "[recurrent][elman][forward]") {
        GIVEN("an Elman network with a hidden unit and a context unit") {
            Algorithm algorithm(0.01f, elmanConnections());
            configureAnalyticWeights(algorithm);
            ReferenceElman reference;

            WHEN("a sequence of inputs is evaluated one timestep at a time") {
                const std::array< float, 5 > sequence{2.f, 6.f, -4.f, 0.f, 3.f};

                THEN("every output matches the recurrence equation") {
                    for(const float value : sequence) {
                        const std::array< Input, 2 > inputs{Input{value}, Input{value}};
                        std::array< float, 1 > output{};
                        algorithm.evaluate(inputs.begin(), inputs.end(), output.begin());
                        REQUIRE_THAT(output[0],
                                     Catch::Matchers::WithinAbs(reference.step(value), 1e-5f));
                    }
                }
            }
        }
    }

    SCENARIO("Elman network: state persists and can be cleared",
             "[recurrent][elman][state]") {
        GIVEN("a recurrent state after the first evaluation") {
            Algorithm algorithm(0.01f, elmanConnections());
            configureAnalyticWeights(algorithm);

            const std::array< Input, 2 > highInputs{Input{6.f}, Input{6.f}};
            const std::array< Input, 2 > lowInputs{Input{-6.f}, Input{-6.f}};
            std::array< float, 1 > first{};
            std::array< float, 1 > second{};
            std::array< float, 1 > restarted{};

            algorithm.evaluate(highInputs.begin(), highInputs.end(), first.begin());
            algorithm.evaluate(lowInputs.begin(), lowInputs.end(), second.begin());

            WHEN(
             "the hidden state is cleared and the first input is repeated") {
                std::get< 1 >(algorithm.context().outputs).fill(0.f);
                algorithm.evaluate(highInputs.begin(),
                                   highInputs.end(),
                                   restarted.begin());

                THEN("the second output depends on the previous timestep") {
                    REQUIRE_THAT(second[0], !Catch::Matchers::WithinAbs(first[0], 1e-3f));
                }

                THEN("clearing the state reproduces the first output") {
                    REQUIRE_THAT(restarted[0], Catch::Matchers::WithinAbs(first[0], 1e-5f));
                }
            }
        }
    }

    SCENARIO("Elman network: training reads the previous input from context",
             "[recurrent][elman][training]") {
        GIVEN("a delayed copy sequence with a fixed hidden encoding") {
            auto connections = elmanConnections();
            connections.set< 0, 0 >(2, Connection{});
            Algorithm algorithm(0.8f, connections);
            configureReadoutWeights(algorithm);

            const std::array< bool, 16 > sequence{
             false, true, false, true, true, true, false, false, true, false, true, true, false, true, true, true};

            WHEN("the readout is trained on consecutive timestep pairs") {
                const auto reservoirWeights = std::get< 1 >(algorithm.context().weights);
                const auto reservoirBiases = std::get< 1 >(algorithm.context().biases);
                for(int epoch = 0; epoch < 100; ++epoch) {
                    std::get< 1 >(algorithm.context().outputs).fill(0.f);
                    for(std::size_t i = 1; i < sequence.size(); ++i) {
                        const auto prototype =
                         makeDelayedCopyPrototype(sequence[i - 1], sequence[i]);
                        algorithm.executeTrainingStep(prototype, [](float, float next) {
                            return next;
                        });
                        std::get< 1 >(algorithm.context().weights) = reservoirWeights;
                        std::get< 1 >(algorithm.context().biases) = reservoirBiases;
                    }
                }

                THEN("each timestep reproduces the preceding input") {
                    std::get< 1 >(algorithm.context().outputs).fill(0.f);
                    for(std::size_t i = 1; i < sequence.size(); ++i) {
                        const auto prototype =
                         makeDelayedCopyPrototype(sequence[i - 1], sequence[i]);
                        std::array< float, 1 > output{};
                        algorithm.evaluate(std::get< 0 >(prototype).begin(),
                                           std::get< 0 >(prototype).end(),
                                           output.begin());

                        const bool expected = sequence[i - 1];
                        REQUIRE(output[0] > (expected ? 0.8f : -1.f));
                        REQUIRE(output[0] < (expected ? 2.f : 0.2f));
                    }
                }
            }
        }
    }

} // namespace
