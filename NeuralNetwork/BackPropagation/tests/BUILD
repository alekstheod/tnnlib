cc_test(
    name = "tests",
    srcs = glob(
        ["*.cpp"],
        exclude = ["BPOpenCLNeuralLayerTest.cpp", "BPConvolutionNeuralLayerTest.cpp", "OptimizerTest.cpp"],
    ),
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/BackPropagation",
        "//NeuralNetwork/NeuralLayer",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "optimizer_test",
    srcs = ["OptimizerTest.cpp"],
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/BackPropagation",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "bp_opencl_layer_ut",
    srcs = ["BPOpenCLNeuralLayerTest.cpp"],
    tags = ["openCL"],
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/BackPropagation",
        "//NeuralNetwork/BackPropagation:OpenCLNeuralLayer",
        "//NeuralNetwork/NeuralLayer:OpenCLNeuralLayer",
        "//NeuralNetwork/Neuron",
        "@catch2//:catch2_main",
    ],
)
