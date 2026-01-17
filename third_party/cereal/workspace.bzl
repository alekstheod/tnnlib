load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def init_cereal_repo():
    new_git_repository(
        name = "cereal",
        build_file = "//third_party/cereal:BUILD",
        commit = "02eace19a99ce3cd564ca4e379753d69af08c2c8",
        remote = "https://github.com/USCiLab/cereal",
    )