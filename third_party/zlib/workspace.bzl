load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def init_zlib_repo():
    new_git_repository(
        name = "zlib",
        build_file = "//third_party/zlib:BUILD",
        commit = "cacf7f1d4e3d44d871b605da3b647f07d718623f",
        remote = "https://github.com/madler/zlib",
    )