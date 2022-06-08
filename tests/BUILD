# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library
cc_test(
    name = "tnnlib_utest",
    srcs = glob([
        "TestConvolutionLayer.cpp",
        "TestNeuron.cpp",
        "main.cpp",
        "etc/**/*.h",
    ]),
    includes = ["include"],
    deps = [
        "//:tnnlib",
        "//:tnnlib_utils",
        "@catch2",
    ],
)

cc_test(
    name = "tnnlib_opencl_utest",
    srcs = glob([
        "TestOpenCLNeuralLayer.cpp",
        "etc/**/*.h",
    ]),
    includes = ["include"],
    deps = [
        "//:tnnlib",
        "//:tnnlib_utils",
        "@OpenCL",
        "@catch2",
    ],
)
