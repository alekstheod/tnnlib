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
        "*.cpp",
        "etc/**/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":catch2",
        "//src:tnnlib",
        "//src:tnnlib_utils",
    ],
)
