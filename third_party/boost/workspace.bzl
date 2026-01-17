load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def init_boost_repo():
    # Add required boost modules hermetically using specific versions
    http_archive(
        name = "boost.asio",
        urls = ["https://github.com/bazelbuild/bazel-central-registry/archive/refs/heads/main.zip"],
        strip_prefix = "bazel-central-registry-main/modules/boost.asio/1.89.0.bcr.2",
    )

    http_archive(
        name = "boost.filesystem",
        urls = ["https://github.com/bazelbuild/bazel-central-registry/archive/refs/heads/main.zip"],
        strip_prefix = "bazel-central-registry-main/modules/boost.filesystem/1.89.0.bcr.2",
    )

    http_archive(
        name = "boost.variant", 
        urls = ["https://github.com/bazelbuild/bazel-central-registry/archive/refs/heads/main.zip"],
        strip_prefix = "bazel-central-registry-main/modules/boost.variant/1.89.0.bcr.2",
    )

    http_archive(
        name = "boost.numeric_conversion",
        urls = ["https://github.com/bazelbuild/bazel-central-registry/archive/refs/heads/main.zip"],
        strip_prefix = "bazel-central-registry-main/modules/boost.numeric_conversion/1.89.0.bcr.2",
    )