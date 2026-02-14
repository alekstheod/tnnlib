cc_test(
    name = "test",
    srcs = glob(
        ["*.cpp"],
        exclude = ["OpenCLNeuralLayerTest.cpp"],
    ),
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/NeuralLayer",
        "//NeuralNetwork/Neuron",
        "@catch2//:catch2_main",
    ],
)

# OpenCL tests are enabled with version compatibility fixes
cc_test(
    name = "opencl_utest",
    srcs = ["OpenCLNeuralLayerTest.cpp"],
    includes = ["include"],
    tags = ["openCL"],
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/NeuralLayer:OpenCLNeuralLayer",
        "//NeuralNetwork/Neuron",
        "@catch2//:catch2_main",
    ],
)
