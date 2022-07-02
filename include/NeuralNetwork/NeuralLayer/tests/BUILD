cc_test(
    name = "test",
    srcs = glob(
        [
            "*.cpp",
        ],
        exclude = ["OpenCLNeuralLayerTest.cpp"],
    ),
    deps = [
        "//:tnnlib",
        "//:tnnlib_utils",
        "//Utilities/catch2:catch2_main",
    ],
)

cc_test(
    name = "opencl_utest",
    srcs = glob([
        "OpenCLNeuralLayerTest.cpp",
    ]),
    includes = ["include"],
    tags = ["openCL"],
    deps = [
        "//:tnnlib",
        "//:tnnlib_utils",
        "//Utilities/catch2:catch2_main",
        "@OpenCL",
    ],
)
