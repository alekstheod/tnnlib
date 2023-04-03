cc_library(
    name = "Perceptron",
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
        "//:tnnlib_utils",
        "//NeuralNetwork/Serialization:Memento",
        "//NeuralNetwork/Utils",
    ],
)
