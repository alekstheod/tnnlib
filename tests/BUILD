# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library
cc_test(
    name = "tnnlib_utest",
    srcs = glob([
        "ConvolutionLayerTest.cpp",
        "NeuronTest.cpp",
        "PerceptronTest.cpp",
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
        "OpenCLNeuralLayerTest.cpp",
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
