cc_library(
    name = "Perceptron",
    hdrs = glob(
        ["*.h"],
        exclude = [
            "tests/**",
        ],
    ),
    copts = ["-Werror"],
    includes = ["."],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/Serialization:Memento",
        "//NeuralNetwork/Utils",
    ],
)

cc_test(
    name = "test",
    srcs = [
        "tests/BuilderTest.cpp",
    ],
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/NeuralLayer",
        "//NeuralNetwork/Neuron",
        "//NeuralNetwork/Perceptron",
        "@catch2//:catch2_main",
    ],
)
