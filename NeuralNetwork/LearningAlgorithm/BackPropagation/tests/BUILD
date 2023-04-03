cc_test(
    name = "tests",
    srcs = glob(
        ["*.cpp"],
        exclude = ["BPOpenCLNeuralLayerTest.cpp"],
    ),
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/LearningAlgorithm/BackPropagation",
        "//NeuralNetwork/Neuron",
        "//Utilities/catch2:catch2_main",
    ],
)

cc_test(
    name = "bp_opencl_layer_ut",
    srcs = ["BPOpenCLNeuralLayerTest.cpp"],
    tags = ["openCL"],
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/LearningAlgorithm/BackPropagation",
        "//NeuralNetwork/LearningAlgorithm/BackPropagation:OpenCLNeuralLayer",
        "//NeuralNetwork/NeuralLayer:OpenCLNeuralLayer",
        "//NeuralNetwork/Neuron",
        "//Utilities/catch2:catch2_main",
    ],
)
