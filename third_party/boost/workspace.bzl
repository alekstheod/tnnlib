load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def init_boost_repo():
    # Use a single boost archive instead of individual modules
    http_archive(
        name = "boost",
        build_file = "//third_party/boost:BUILD",
        sha256 = "4d27e9efed0f6f152dc28db6430b9d3dfb40c0345da7342eaa5a987dde57bd95",
        strip_prefix = "boost-1.84.0",
        urls = ["https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.gz"],
    )