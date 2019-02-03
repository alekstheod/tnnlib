load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "git_repository",
    "new_git_repository",
)

git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "6d6fd834281cb8f8e758dd9ad76df86304bf1869",
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
    commit = "eddf9023206dc40974c26f589ee2ad63a4227a1e",
    remote = "https://github.com/glennrp/libpng",
)
