cc_library(
    name = "BackPropagation",
    hdrs = glob(
        ["*.h"],
        exclude = [
            "tests/**",
            "OpenCL/**",
        ],
    ),
    copts = ["-Werror"],
    includes = ["."],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//NeuralNetwork/NeuralLayer",
        "//NeuralNetwork/Perceptron",
        "@range-v3",
    ],
)

cc_library(
    name = "OpenCLNeuralLayer",
    hdrs = glob(
        ["OpenCL/*.h"],
        exclude = [
            "tests/**",
        ],
    ),
    copts = ["-Werror"],
    data = ["OpenCL/calc_weights.cl"],
    includes = ["."],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":BackPropagation",
        "//NeuralNetwork/NeuralLayer:OpenCLNeuralLayer",
        "@range-v3",
    ],
)
