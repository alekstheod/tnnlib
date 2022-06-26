# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library
cc_test(
    name = "test",
    srcs = glob([
        "PerceptronTest.cpp",
    ]),
    data = ["etc/xor.xml"],
    includes = ["include"],
    deps = [
        "//:tnnlib",
        "//:tnnlib_utils",
        "@catch2",
    ],
)
