cc_library(
    name = "tnnlib",
    hdrs = glob(["NeuralNetwork/**/*.h"]),
    copts = ["-Werror"],
    includes = ["NeuralNetwork"],
    strip_include_prefix = "NeuralNetwork",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@cereal",
        "//src/libs:range-v3",
        ":tnnlib_utils",
        "@boost//:array",
        "@boost//:bind",
        "@boost//:iterator",
        #"@boost//:safe_numerics",
    ],
)

cc_binary(
    name = "ocr",
    srcs = glob(["ocr/**/*.cpp"]),
    copts = ["-Werror"],
    deps = [
        ":tnnlib",
        ":tnnlib_utils",
        "@boost//:filesystem",
        "@boost//:gil",
        "@boost//:variant",
        "@cereal",
        "@libpng",
        "@zlib",
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
