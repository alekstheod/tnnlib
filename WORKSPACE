load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "git_repository",
    "new_git_repository",
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
