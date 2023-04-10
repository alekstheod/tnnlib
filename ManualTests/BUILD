# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_binary
cc_test(
    name = "tnnlib_test",
    srcs = glob(["**/*.cpp"]),
    args = ["1000"],
    copts = ["-Werror"],
    deps = [
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/LearningAlgorithm/BackPropagation",
        "//NeuralNetwork/NeuralLayer",
        "//NeuralNetwork/NeuralLayer:OpenCLNeuralLayer",
        "//NeuralNetwork/Neuron",
        "//NeuralNetwork/Serialization",
    ],
)
