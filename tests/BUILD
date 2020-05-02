# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library
cc_library(
    name = "catch2",
    hdrs = ["include/catch.hpp"],
    includes = ["include"],
    strip_include_prefix = "include",
)

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
        ":catch2",
        "//:tnnlib",
        "//:tnnlib_utils",
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
        ":catch2",
        "//:tnnlib",
        "//:tnnlib_utils",
        "@OpenCL",
    ],
)
