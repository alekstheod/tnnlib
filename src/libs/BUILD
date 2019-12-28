# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library

cc_library(
    name = "range-v3",
    hdrs = glob(["range-v3/nix/**/*.hpp"]),
    includes = [
        "range-v3/nix",
    ],
    strip_include_prefix = "range-v3/nix",
    visibility = ["//visibility:public"],
)
