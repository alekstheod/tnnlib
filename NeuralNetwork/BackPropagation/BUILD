load("//:opencl_kernel.bzl", "opencl_kernel", "opencl_kernel_binary")

filegroup(
    name = "opencl_kernels",
    srcs = glob(["OpenCL/*.cl"]),
    visibility = ["//visibility:public"],
)

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

opencl_kernel(
    name = "calc_weights_kernel",
    src = "OpenCL/calc_weights.cl",
)

opencl_kernel_binary(
    name = "calc_weights_kernel_binary",
    src = ":calc_weights_kernel.spv",
    var_name = "calc_weights",
)

cc_library(
    name = "calc_weights_kernel_binary_h",
    hdrs = [":calc_weights_kernel_binary.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
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
    ) + [
        ":calc_weights_kernel_binary.h",
    ],
    copts = ["-Werror"],
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
