#include "NeuralNetwork/BackPropagation/BPContext.h"
#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/BepAlgorithm.h"
#include "NeuralNetwork/Perceptron/PerceptronBuilder.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"

#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS
#include <catch2/catch_all.hpp>

namespace {

    using namespace nn;

    SCENARIO("Connection matrix: builder creates ConnectionData",
             "[connection_matrix][builder]") {
        GIVEN("A builder with connection matrix") {
            auto builder = build<float>()
                .input< 4 >()
                .dense< 4 >()
                .with_connection_matrix< 0 >(0, 1)
                .with_connection_matrix< 1 >(1, 2);

            WHEN("connections() is called") {
                const auto& conn = builder.connections();
                THEN("connection data is stored for the correct layer") {
                    REQUIRE(conn.layers.size() == 2);
                    REQUIRE(conn.layers[0].empty());
                    REQUIRE(conn.layers[1].size() == 2);
                    REQUIRE(conn.layers[1][0].size() == 2);
                    REQUIRE(conn.layers[1][0][0] == 0);
                    REQUIRE(conn.layers[1][0][1] == 1);
                    REQUIRE(conn.layers[1][1].size() == 2);
                    REQUIRE(conn.layers[1][1][0] == 1);
                    REQUIRE(conn.layers[1][1][1] == 2);
                }
            }
        }
    }

    SCENARIO("Connection matrix: default connections are fully connected",
             "[connection_matrix][bp_context]") {
        GIVEN("A BPContext with default connections") {
            using Layer = NeuralLayer< Neuron, SigmoidFunction, 4, 4 >;
            using BPCtx = bp::BPContext< float, std::tuple< Layer > >;
            BPCtx ctx;

            auto& conn = std::get< 0 >(ctx.connections);
            utils::for_< 4 >([&](auto i) {
                for(std::size_t j = 0; j < 4; ++j) {
                    REQUIRE(conn[i.value][j] == j);
                }
            });
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
            connData.add< 0 >(1, 0, 1);
            connData.add< 1 >(1, 1, 2);

            WHEN("BepAlgorithm is constructed with connection data") {
                bp::BepAlgorithm< Perceptron > algo(0.01f, connData);

                THEN("overridden neurons have custom connections, others have defaults") {
                    const auto& ctx = algo.context();
                    const auto& layerConn = std::get< 1 >(ctx.connections);

                    REQUIRE(layerConn[0][0] == 0);
                    REQUIRE(layerConn[0][1] == 1);
                    REQUIRE(layerConn[0][2] == bp::connectionSentinel);
                    REQUIRE(layerConn[0][3] == bp::connectionSentinel);

                    REQUIRE(layerConn[1][0] == 1);
                    REQUIRE(layerConn[1][1] == 2);
                    REQUIRE(layerConn[1][2] == bp::connectionSentinel);
                    REQUIRE(layerConn[1][3] == bp::connectionSentinel);

                    REQUIRE(layerConn[2][0] == 0);
                    REQUIRE(layerConn[2][1] == 1);
                    REQUIRE(layerConn[2][2] == 2);
                    REQUIRE(layerConn[2][3] == 3);

                    REQUIRE(layerConn[3][0] == 0);
                    REQUIRE(layerConn[3][1] == 1);
                    REQUIRE(layerConn[3][2] == 2);
                    REQUIRE(layerConn[3][3] == 3);
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
            connData.add< 0 >(1, 0);

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

} // namespace
