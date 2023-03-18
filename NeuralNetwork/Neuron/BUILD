cc_library(
    name = "Neuron",
    hdrs = glob(
        ["*.h"],
        exclude = [
            "tests/**",
        ],
    ),
    copts = ["-Werror"],
    includes = ["."],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//:tnnlib_utils",
    ],
)
