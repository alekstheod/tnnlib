load("//:build_defs.bzl", "if_rocm", "rocm_library")

rocm_library(
    name = "gpu_train_kernel",
    srcs = ["src/gpu/train_kernel.hip.cpp"],
    copts = ["-Werror"],
    deps = [
        ":gpu_train_support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gpu_train_support",
    srcs = ["src/gpu/train.cpp"],
    hdrs = ["src/gpu/train.h"],
    copts = ["-Werror"],
    deps = [
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/BackPropagation",
        "//NeuralNetwork/Neuron",
        "//NeuralNetwork/Perceptron",
    ] + if_rocm(["@config_rocm_hipcc//rocm:hip_runtime"]),
    alwayslink = 1,
)

cc_library(
    name = "gpu_train",
    deps = select({
        "//:using_rocm": [":gpu_train_kernel"],
        "//conditions:default": [":gpu_train_support"],
    }),
    alwayslink = 1,
)

rocm_library(
    name = "list_devices_lib",
    srcs = ["src/gpu/list_devices.hip.cpp"],
    copts = ["-Werror"],
    alwayslink = 1,
)

cc_binary(
    name = "list_devices",
    deps = [":list_devices_lib"] + if_rocm(["@config_rocm_hipcc//rocm:hip_runtime"]),
)

cc_binary(
    name = "ocr",
    srcs = glob(["src/*.cpp"]),
    copts = ["-Werror"],
    data = [
        "etc/perceptron.json",
        "@OcrSamples//:samples",
    ],
    deps = [
        ":gpu_train",
        "//:tnnlib_utils",
        "//NeuralNetwork/ActivationFunction",
        "//NeuralNetwork/BackPropagation",
        "//NeuralNetwork/BackPropagation:OpenCLNeuralLayer",
        "//NeuralNetwork/NeuralLayer:OpenCLNeuralLayer",
        "//NeuralNetwork/Neuron",
        "//NeuralNetwork/Serialization",
        "//third_party/opencl:opencl",
        "@boost//:asio",
        "@boost//:filesystem",
        "@boost//:gil",
        "@boost//:variant",
        "@cereal",
        "@libpng",
        "@zlib",
    ],
)
