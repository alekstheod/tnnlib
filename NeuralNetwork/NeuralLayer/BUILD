load("//:opencl_kernel.bzl", "opencl_kernel", "opencl_kernel_binary")

filegroup(
    name = "opencl_kernels",
    srcs = glob(["OpenCL/*.cl"]),
    visibility = ["//visibility:public"],
)

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
        "//NeuralNetwork/Neuron",
        "//NeuralNetwork/Serialization:Memento",
        "@boost//:asio",
        "@cereal",
        "@range-v3",
    ],
)

opencl_kernel(
    name = "dot_product_kernel",
    src = "OpenCL/dot_product.cl",
)

opencl_kernel_binary(
    name = "dot_product_kernel_binary",
    src = ":dot_product_kernel.spv",
    var_name = "dot_product",
)

cc_library(
    name = "dot_product_kernel_binary_h",
    hdrs = [":dot_product_kernel_binary.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
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
    includes = ["."],
    deps = [
        ":NeuralLayer",
        ":dot_product_kernel_binary_h",
        "//:tnnlib_utils",
        "//third_party/opencl:opencl",
    ],
    tags = ["openCL"],
    visibility = [
        "//visibility:public",
    ],
)
