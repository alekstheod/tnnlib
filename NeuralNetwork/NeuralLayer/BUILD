cc_library(
    name = "NeuralLayer",
    srcs = glob(
        ["**/*.cpp"],
        exclude = [
            "tests/**",
            "OpenCLNeuralLayer.h",
            "OpenCLNeuralLayer.cpp",
        ],
    ),
    hdrs = glob(
        ["*.h"],
        exclude = [
            "tests/**",
            "OpenCLNeuralLayer.h",
            "OpenCLNeuralLayer.cpp",
        ],
    ),
    copts = ["-Werror"],
    includes = ["."],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//:tnnlib_utils",
        "//NeuralNetwork/Serialization",
        "@boost//:asio",
        "@cereal",
        "@range-v3",
    ],
)

cc_library(
    name = "OpenCLNeuralLayer",
    srcs = [
        "OpenCLNeuralLayer.cpp",
    ],
    hdrs = [
        "OpenCLNeuralLayer.h",
    ],
    copts = ["-Werror"],
    data = ["dot_product.cl"],
    includes = ["."],
    tags = ["openCL"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":NeuralLayer",
        "//:tnnlib_utils",
        "@OpenCL",
    ],
)
