# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library

cc_library(
    name = "libpng_config",
    hdrs = ["libpng/pnglibconf.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "range-v3",
    hdrs = glob(["range-v3/nix/**/*.hpp"]),
    includes = [
        "range-v3/nix",
    ],
    strip_include_prefix = "range-v3/nix",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cereal",
    hdrs = glob(["cereal/**/*.hpp"]),
    includes = ["cereal"],
    strip_include_prefix = ".",
    visibility = ["//visibility:public"],
)
