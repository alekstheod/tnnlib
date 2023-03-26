cc_library(
    name = "BackPropagation",
    srcs = glob(
        ["**/*.cpp"],
        exclude = [
            "tests/**",
            "OpenCL/**",
        ],
    ),
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
    srcs = glob(
        ["OpenCL/*.cpp"],
        exclude = [
            "tests/**",
        ],
    ),
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
