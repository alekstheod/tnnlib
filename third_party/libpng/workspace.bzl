load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def init_libpng_repo():
    new_git_repository(
        name = "libpng",
        build_file = "//third_party/libpng:BUILD",
        commit = "c17d164b4467f099b4484dfd4a279da0bc1dbd4a",
        remote = "https://github.com/glennrp/libpng",
    )