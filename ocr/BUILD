cc_binary(
    name = "ocr",
    srcs = glob(["src/*.cpp"]),
    copts = ["-Werror"],
    data = glob(["samples/**"]),
    deps = [
        "//:tnnlib",
        "//:tnnlib_utils",
        "@OpenCL",
        "@boost//:gil",
        "@boost//:variant",
        "@cereal",
        "@libpng",
        "@zlib",
    ],
)
