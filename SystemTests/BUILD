# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library
cc_test(
    name = "test",
    srcs = glob([
        "PerceptronTest.cpp",
    ]),
    data = ["etc/xor.json"],
    includes = ["include"],
    tags = ["system"],
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/LearningAlgorithm/BackPropagation",
        "//NeuralNetwork/NeuralLayer",
        "//NeuralNetwork/Neuron",
        "//NeuralNetwork/Serialization",
        "@catch2",
    ],
)
