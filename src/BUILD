cc_library(
    name = "tnnlib",
    hdrs = glob(["NeuralNetwork/**/*.h"]),
    includes = ["NeuralNetwork"],
    strip_include_prefix = "NeuralNetwork",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//src/libs:cereal",
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
    copts = [],
    deps = [
        ":tnnlib",
        ":tnnlib_utils",
        "//src/libs:cereal",
        "//src/libs:libpng_config",
        "@boost//:filesystem",
        "@libpng",
        "@zlib",
    ],
)

cc_library(
    name = "tnnlib_utils",
    srcs = glob(["Utilities/**/*.cpp"]),
    hdrs = glob(["Utilities/**/*.h"]),
    includes = ["Utilities"],
    strip_include_prefix = "Utilities",
    visibility = ["//visibility:public"],
)
