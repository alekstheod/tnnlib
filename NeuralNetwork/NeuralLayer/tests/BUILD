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
        "//Utilities/catch2:catch2_main",
    ],
)

cc_test(
    name = "opencl_utest",
    srcs = glob([
        "OpenCLNeuralLayerTest.cpp",
    ]),
    includes = ["include"],
    tags = ["openCL"],
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/NeuralLayer:OpenCLNeuralLayer",
        "//NeuralNetwork/Neuron",
        "//Utilities/catch2:catch2_main",
        "@OpenCL",
    ],
)
