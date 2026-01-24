load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def init_range_v3_repo():
    new_git_repository(
        name = "range-v3",
        build_file = "//third_party/range-v3:BUILD",
        commit = "a81477931a8aa2ad025c6bda0609f38e09e4d7ec",
        remote = "https://github.com/ericniebler/range-v3",
    )