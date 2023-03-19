cc_library(
    name = "NeuralLayer",
    srcs = glob(
        ["**/*.cpp"],
        exclude = [
            "tests/**",
            "OpenCL/**",
        ],
    ),
    hdrs = glob(
        ["**/*.h"],
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
        "OpenCL/OpenCLNeuralLayer.cpp",
    ],
    hdrs = [
        "OpenCL/OpenCLNeuralLayer.h",
    ],
    copts = ["-Werror"],
    data = ["OpenCL/dot_product.cl"],
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
