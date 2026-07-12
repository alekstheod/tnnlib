load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

config_setting(
    name = "using_rocm",
    values = {"define": "using_rocm=true"},
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

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
        "//ocr:ocr": "",
    },
)
