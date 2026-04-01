cc_test(
    name = "builder_tests",
    srcs = glob(["*.cpp"]),
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/NeuralLayer",
        "//NeuralNetwork/NeuralLayer:OpenCLNeuralLayer",
        "//NeuralNetwork/Neuron",
        "//NeuralNetwork/Perceptron",
        "@catch2//:catch2_main",
    ],
)
