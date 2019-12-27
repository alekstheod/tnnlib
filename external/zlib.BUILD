licenses(["notice"])

cc_library(
    name = "zlib",
    srcs = glob(
        [
            "**/*.c",
            "**/*.h",
        ],
        exclude = [
            "zlib.h",
            "contrib/**/*",
        ],
    ),
    hdrs = ["zlib.h"],
    copts = [
        "-Wno-implicit-function-declaration",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
