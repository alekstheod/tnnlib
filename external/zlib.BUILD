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
            "test/**/*",
        ],
    ),
    hdrs = ["zlib.h"],
    copts = [
        "-w",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
