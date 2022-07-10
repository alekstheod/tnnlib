cc_library(
    name = "tnnlib",
    hdrs = glob(
        ["include/NeuralNetwork/**/*.h"],
        exclude = ["**/tests/**"],
    ),
    copts = ["-Werror"],
    includes = ["NeuralNetwork"],
    strip_include_prefix = "include",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":tnnlib_utils",
        "@boost//:array",
        "@boost//:bind",
        "@boost//:iterator",
        "@cereal",
        "@range-v3",
    ],
)

cc_library(
    name = "tnnlib_utils",
    srcs = glob(["Utilities/**/*.cpp"]),
    hdrs = glob(["Utilities/**/*.h"]),
    copts = ["-Werror"],
    includes = ["Utilities"],
    strip_include_prefix = "Utilities",
    visibility = ["//visibility:public"],
    deps = ["@boost//:numeric_conversion"],
)
