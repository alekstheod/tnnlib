cc_library(
    name = "BackPropagation",
    srcs = glob(
        ["**/*.cpp"],
        exclude = [
            "tests/**",
        ],
    ),
    hdrs = glob(
        ["*.h"],
        exclude = [
            "tests/**",
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
