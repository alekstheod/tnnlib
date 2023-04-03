cc_library(
    name = "Serialization",
    hdrs = ["Cereal.h"],
    copts = ["-Werror"],
    includes = ["."],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":Memento",
        "@cereal",
        "@range-v3",
    ],
)

cc_library(
    name = "Memento",
    hdrs = glob(
        ["*.h"],
        exclude = [
            "tests/**",
            "Cereal.h",
        ],
    ),
    copts = ["-Werror"],
    includes = ["."],
    visibility = [
        "//visibility:public",
    ],
)
