load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "git_repository",
    "new_git_repository",
)
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

local_repository(
    name = "libpng_config",
    path = "external/libpng_config",
)

http_archive(
    name = "OcrSamples",
    build_file = "samples.BUILD",
    sha256 = "165a7e4e56a2307cb74e212c83456627f5ab91516a35c51114626f488b96da77",
    urls = ["https://github.com/alekstheod/tnnlib/raw/master/ocr/samples.zip"],
)

http_archive(
    name = "catch2",
    sha256 = "b9b592bd743c09f13ee4bf35fc30eeee2748963184f6bea836b146e6cc2a585a",
    strip_prefix = "Catch2-2.13.8",
    urls = ["https://github.com/catchorg/Catch2/archive/v2.13.8.tar.gz"],
)

new_local_repository(
    name = "OpenCL",
    build_file = "external/OpenCL.BUILD",
    path = "/usr/",
)

git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "ed844db5990d21b75dc3553c057069f324b3916b",
    remote = "https://github.com/nelhage/rules_boost",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

new_git_repository(
    name = "zlib",
    build_file = "zlib.BUILD",
    commit = "cacf7f1d4e3d44d871b605da3b647f07d718623f",
    remote = "https://github.com/madler/zlib",
)

new_git_repository(
    name = "libpng",
    build_file = "libpng.BUILD",
    commit = "c17d164b4467f099b4484dfd4a279da0bc1dbd4a",
    remote = "https://github.com/glennrp/libpng",
)

new_git_repository(
    name = "cereal",
    build_file = "cereal.BUILD",
    commit = "02eace19a99ce3cd564ca4e379753d69af08c2c8",
    remote = "https://github.com/USCiLab/cereal",
)

new_git_repository(
    name = "range-v3",
    build_file = "range-v3.BUILD",
    commit = "a81477931a8aa2ad025c6bda0609f38e09e4d7ec",
    remote = "https://github.com/ericniebler/range-v3",
)
