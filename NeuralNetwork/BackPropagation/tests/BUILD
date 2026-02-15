cc_test(
    name = "tests",
    srcs = glob(
        ["*.cpp"],
        exclude = ["BPOpenCLNeuralLayerTest.cpp"],
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
    name = "bp_opencl_layer_ut",
    srcs = ["BPOpenCLNeuralLayerTest.cpp"],
    tags = ["openCL"],
    data = [
        "//NeuralNetwork/NeuralLayer:opencl_kernels",
        "//NeuralNetwork/BackPropagation:opencl_kernels",
    ],
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
