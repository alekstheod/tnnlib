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
