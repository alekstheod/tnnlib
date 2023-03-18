cc_library(
    name = "Serialization",
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
        "//NeuralNetwork/Neuron",
        "@cereal",
        "@range-v3",
    ],
)
