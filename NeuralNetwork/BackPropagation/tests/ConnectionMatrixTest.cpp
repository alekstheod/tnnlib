#include "NeuralNetwork/BackPropagation/BPContext.h"
#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/BepAlgorithm.h"
#include "NeuralNetwork/Perceptron/PerceptronBuilder.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"

#include <cereal/archives/json.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>

#include <cmath>
#include <sstream>

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch_all.hpp>

namespace {

    using namespace nn;

    float sigmoid(float x) {
        return 1.f / (1.f + std::exp(-x));
    }

    SCENARIO("Connection matrix: builder creates ConnectionData",
             "[connection_matrix][builder]") {
        GIVEN("A builder with connection matrix") {
            auto builder = build<float>()
                .input< 4 >()
                .dense< 4 >()
                .with_connection_matrix< 0 >(connect(0, 0), connect(0, 1))
                .with_connection_matrix< 1 >(connect(0, 1), connect(0, 2));

            WHEN("connections() is called") {
                const auto& conn = builder.connections();
                THEN("connection data is stored for the correct layer") {
                    REQUIRE(conn.layers.size() == 2);
                    REQUIRE(conn.layers[0].empty());
                    REQUIRE(conn.layers[1].size() == 2);
                    REQUIRE(conn.layers[1][0].size() == 2);
                    REQUIRE(conn.layers[1][0][0] == Connection{0, 0});
                    REQUIRE(conn.layers[1][0][1] == Connection{0, 1});
                    REQUIRE(conn.layers[1][1].size() == 2);
                    REQUIRE(conn.layers[1][1][0] == Connection{0, 1});
                    REQUIRE(conn.layers[1][1][1] == Connection{0, 2});
                }
            }
        }
    }

    SCENARIO("Connection matrix: default connections wire the predecessor",
             "[connection_matrix][bp_context]") {
        GIVEN("A BPContext with two layers") {
            using InputL = NeuralLayer< Neuron, SigmoidFunction, 4 >;
            using HiddenL = NeuralLayer< Neuron, SigmoidFunction, 2, 4 >;
            using BPCtx = bp::BPContext< float, std::tuple< InputL, HiddenL > >;
            BPCtx ctx;

            THEN("the input layer has no sources and the hidden layer reads its predecessor") {
                auto& inputConn = std::get< 0 >(ctx.connections);
                utils::for_< 4 >([&](auto i) {
                    for(std::size_t s = 0; s < 4; ++s) {
                        REQUIRE_FALSE(inputConn[i.value][s].is_connected());
                    }
                });

                auto& hiddenConn = std::get< 1 >(ctx.connections);
                utils::for_< 2 >([&](auto i) {
                    for(std::size_t s = 0; s < 4; ++s) {
                        REQUIRE(hiddenConn[i.value][s] == Connection{0, s});
                    }
                });
            }
        }
    }

    SCENARIO("Connection matrix: BepAlgorithm initializes connections",
             "[connection_matrix][bep_algorithm]") {
        GIVEN("A perceptron and connection data") {
            using Perceptron = decltype(
                build<float>()
                    .input< 4 >()
                    .dense< 4 >()
                    .dense< 2 >()
            )::type;

            bp::ConnectionData connData;
            connData.add< 0 >(1, connect(0, 0), connect(0, 1));
            connData.add< 1 >(1, connect(0, 1), connect(0, 2));

            WHEN("BepAlgorithm is constructed with connection data") {
                bp::BepAlgorithm< Perceptron > algo(0.01f, connData);

                THEN("overridden neurons have custom connections, others have defaults") {
                    const auto& ctx = algo.context();
                    const auto& layerConn = std::get< 1 >(ctx.connections);

                    REQUIRE(layerConn[0][0] == Connection{0, 0});
                    REQUIRE(layerConn[0][1] == Connection{0, 1});
                    REQUIRE_FALSE(layerConn[0][2].is_connected());
                    REQUIRE_FALSE(layerConn[0][3].is_connected());

                    REQUIRE(layerConn[1][0] == Connection{0, 1});
                    REQUIRE(layerConn[1][1] == Connection{0, 2});
                    REQUIRE_FALSE(layerConn[1][2].is_connected());
                    REQUIRE_FALSE(layerConn[1][3].is_connected());

                    REQUIRE(layerConn[2][0] == Connection{0, 2});
                    REQUIRE(layerConn[2][1] == Connection{0, 3});
                    REQUIRE(layerConn[3][0] == Connection{0, 3});
                }
            }
        }
    }

    SCENARIO("Connection matrix: training respects connections",
             "[connection_matrix][training]") {
        GIVEN("A simple network where neuron 0 only sees input 0") {
            using Perceptron = decltype(
                build<float>()
                    .input< 2 >()
                    .dense< 1 >()
            )::type;

            bp::ConnectionData connData;
            connData.add< 0 >(1, connect(0, 0));

            bp::BepAlgorithm< Perceptron > algo(0.1f, connData);

            auto& ctx = algo.context();
            float w0_before = std::get< 1 >(ctx.weights)[0];
            float w1_before = std::get< 1 >(ctx.weights)[1];

            WHEN("training step is executed") {
                using Input = Perceptron::Input;
                bp::BepAlgorithm< Perceptron >::Prototype proto{
                    {Input{1.0f}, Input{1.0f}}, {1.0f}};
                algo.executeTrainingStep(proto, [](float, float new_val) { return new_val; });

                THEN("connection 1 weight is unchanged while connection 0 is updated") {
                    float w0_after = std::get< 1 >(ctx.weights)[0];
                    float w1_after = std::get< 1 >(ctx.weights)[1];
                    REQUIRE(w0_after != w0_before);
                    REQUIRE(w1_after == w1_before);
                }
            }
        }
    }

    SCENARIO("Connection matrix: a neuron can read a non-predecessor layer",
             "[connection_matrix][forward]") {
        GIVEN("A network where the output layer reads the input layer directly") {
            using Perceptron = decltype(
                build<float>()
                    .input< 2 >()
                    .dense< 2 >()
                    .dense< 1 >()
            )::type;

            bp::ConnectionData connData;
            connData.add< 0 >(2, connect(0, 1));

            bp::BepAlgorithm< Perceptron > algo(0.1f, connData);

            auto& ctx = algo.context();
            auto& hiddenWeights = std::get< 1 >(ctx.weights);
            auto& hiddenBiases = std::get< 1 >(ctx.biases);
            auto& outputWeights = std::get< 2 >(ctx.weights);
            auto& outputBiases = std::get< 2 >(ctx.biases);

            for(auto& w : hiddenWeights) {
                w = 0.f;
            }
            for(auto& b : hiddenBiases) {
                b = 0.f;
            }
            outputWeights[0] = 1.f;
            outputWeights[1] = 99.f;
            outputBiases[0] = 0.f;

            WHEN("the network is evaluated") {
                using Input = Perceptron::Input;
                std::array< Input, 2 > inputs{Input{1.0f}, Input{1.0f}};
                std::array< float, 1 > output{};
                algo.evaluate(inputs.begin(), inputs.end(), output.begin());

                THEN("the output follows the skip connection, not the hidden layer") {
                    const float expected = sigmoid(sigmoid(1.0f));
                    REQUIRE_THAT(output[0], Catch::Matchers::WithinRel(expected));
                }
            }
        }
    }

    SCENARIO("Connection matrix: same-layer connection reads the previous evaluation",
             "[connection_matrix][recurrent]") {
        GIVEN("A network where neuron 1 reads neuron 0 of its own layer") {
            using Perceptron = decltype(
                build<float>()
                    .input< 1 >()
                    .dense< 2 >()
            )::type;

            bp::ConnectionData connData;
            connData.add< 1 >(1, connect(1, 0));

            bp::BepAlgorithm< Perceptron > algo(0.1f, connData);

            auto& ctx = algo.context();
            auto& weights = std::get< 1 >(ctx.weights);
            auto& biases = std::get< 1 >(ctx.biases);
            weights[0] = 1.f;
            weights[1] = 1.f;
            biases[0] = 0.f;
            biases[1] = 0.f;

            WHEN("the network is evaluated twice") {
                using Input = Perceptron::Input;
                std::array< Input, 1 > inputs{Input{1.0f}};
                std::array< float, 2 > first{};
                std::array< float, 2 > second{};
                algo.evaluate(inputs.begin(), inputs.end(), first.begin());
                algo.evaluate(inputs.begin(), inputs.end(), second.begin());

                THEN("the recurrent neuron picks up the state of the previous pass") {
                    const float neuron0 = sigmoid(sigmoid(1.0f));
                    REQUIRE_THAT(first[0], Catch::Matchers::WithinRel(neuron0));
                    REQUIRE_THAT(first[1], Catch::Matchers::WithinRel(sigmoid(0.f)));
                    REQUIRE_THAT(second[0], Catch::Matchers::WithinRel(neuron0));
                    REQUIRE_THAT(second[1], Catch::Matchers::WithinRel(sigmoid(neuron0)));
                }
            }
        }
    }

    SCENARIO("Connection matrix: only customized neurons are collected",
             "[connection_matrix][serialization]") {
        GIVEN("A network with a single customized neuron") {
            using Perceptron = decltype(
                build<float>()
                    .input< 2 >()
                    .dense< 2 >()
                    .dense< 1 >()
            )::type;

            bp::ConnectionData connData;
            connData.add< 0 >(2, connect(0, 1));

            bp::BepAlgorithm< Perceptron > algo(0.1f, connData);

            WHEN("connectionOverrides() is collected") {
                const auto overrides = algo.context().connectionOverrides();

                THEN("only the customized neuron carries an entry") {
                    REQUIRE(overrides.layers.size() == 3);
                    REQUIRE(overrides.layers[0].empty());
                    REQUIRE(overrides.layers[1].empty());
                    REQUIRE(overrides.layers[2].size() == 1);
                    REQUIRE(overrides.layers[2][0].size() == 2);
                    REQUIRE(overrides.layers[2][0][0] == Connection{0, 1});
                }

                THEN("reapplying the entries over regular connections reproduces the matrix") {
                    auto copy = algo.context();
                    copy.resetConnections();
                    copy.applyConnections(overrides);
                    REQUIRE(copy.connections == algo.context().connections);
                }
            }
        }
    }

    SCENARIO("Connection matrix: sparse entries survive a serialization roundtrip",
             "[connection_matrix][serialization]") {
        GIVEN("A customized algorithm and a plain one") {
            using Perceptron = decltype(
                build<float>()
                    .input< 2 >()
                    .dense< 2 >()
                    .dense< 1 >()
            )::type;

            bp::ConnectionData connData;
            connData.add< 0 >(2, connect(0, 1));

            bp::BepAlgorithm< Perceptron > trained(0.1f, connData);
            bp::BepAlgorithm< Perceptron > reloaded(0.1f);

            std::stringstream stream;
            {
                cereal::JSONOutputArchive oa(stream);
                oa(cereal::make_nvp("context", trained.context()));
            }

            WHEN("the context is loaded") {
                {
                    cereal::JSONInputArchive ia(stream);
                    ia(cereal::make_nvp("context", reloaded.context()));
                }

                THEN("the customized neuron is restored from the entries") {
                    const auto& outputConn = std::get< 2 >(reloaded.context().connections);
                    REQUIRE(outputConn[0][0] == Connection{0, 1});
                    REQUIRE_FALSE(outputConn[0][1].is_connected());
                }

                THEN("neurons without entries fall back to regular connections") {
                    const auto& hiddenConn = std::get< 1 >(reloaded.context().connections);
                    REQUIRE(hiddenConn[0][0] == Connection{0, 0});
                    REQUIRE(hiddenConn[1][0] == Connection{0, 1});
                    const auto& inputConn = std::get< 0 >(reloaded.context().connections);
                    REQUIRE_FALSE(inputConn[0][0].is_connected());
                }

                THEN("the loaded matrix matches the saved one") {
                    REQUIRE(reloaded.context().connections == trained.context().connections);
                }
            }
        }
    }

} // namespace
