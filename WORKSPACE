load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "git_repository",
    "new_git_repository",
)
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/llvm:llvm_workspace.bzl", "init_llvm_repo")

init_llvm_repo()

local_repository(
    name = "libpng_config",
    path = "third_party/libpng_config",
)

http_archive(
    name = "OcrSamples",
    build_file = "//third_party:samples.BUILD",
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
    build_file = "//third_party/opencl:OpenCL.BUILD",
    path = "/usr/",
)

http_archive(
    name = "boost.asio",
    strip_prefix = "bazel-central-registry-main/modules/boost.asio/1.89.0.bcr.2",
    urls = ["https://github.com/bazelbuild/bazel-central-registry/archive/refs/heads/main.zip"],
)

http_archive(
    name = "boost.filesystem",
    strip_prefix = "bazel-central-registry-main/modules/boost.filesystem/1.89.0.bcr.2",
    urls = ["https://github.com/bazelbuild/bazel-central-registry/archive/refs/heads/main.zip"],
)

http_archive(
    name = "boost.variant",
    strip_prefix = "bazel-central-registry-main/modules/boost.variant/1.89.0.bcr.2",
    urls = ["https://github.com/bazelbuild/bazel-central-registry/archive/refs/heads/main.zip"],
)

http_archive(
    name = "boost.numeric_conversion",
    strip_prefix = "bazel-central-registry-main/modules/boost.numeric_conversion/1.89.0.bcr.2",
    urls = ["https://github.com/bazelbuild/bazel-central-registry/archive/refs/heads/main.zip"],
)

new_local_repository(
    name = "boost",
    build_file = "//third_party/boost:BUILD_aliases",
    path = ".",
)

load("//third_party/zlib:workspace.bzl", "init_zlib_repo")

init_zlib_repo()

new_git_repository(
    name = "libpng",
    build_file = "//third_party:libpng.BUILD",
    commit = "c17d164b4467f099b4484dfd4a279da0bc1dbd4a",
    remote = "https://github.com/glennrp/libpng",
)

load("//third_party/cereal:workspace.bzl", "init_cereal_repo")

init_cereal_repo()

load("//third_party/range-v3:workspace.bzl", "init_range_v3_repo")

init_range_v3_repo()

http_archive(
    name = "hedron_compile_commands",
    strip_prefix = "bazel-compile-commands-extractor-0e990032f3c5a866e72615cf67e5ce22186dcb97",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/0e990032f3c5a866e72615cf67e5ce22186dcb97.tar.gz",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()

load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")

hedron_compile_commands_setup_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive_transitive()
