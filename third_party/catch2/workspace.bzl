load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def init_catch2_repo():
    http_archive(
        name = "catch2",
        sha256 = "b9b592bd743c09f13ee4bf35fc30eeee2748963184f6bea836b146e6cc2a585a",
        strip_prefix = "Catch2-2.13.8",
        urls = ["https://github.com/catchorg/Catch2/archive/v2.13.8.tar.gz"],
    )