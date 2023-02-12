cc_binary(
    name = "ocr",
    srcs = glob(["src/*.cpp"]),
    copts = ["-Werror"],
    data = glob(["samples/**"]),
    tags = ["openCL"],
    deps = [
        "//:tnnlib",
        "//:tnnlib_utils",
        "@OpenCL",
        "@boost//:filesystem",
        "@boost//:gil",
        "@boost//:variant",
        "@cereal",
        "@libpng",
        "@zlib",
    ],
)
