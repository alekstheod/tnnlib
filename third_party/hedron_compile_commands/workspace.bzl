load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")
load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")
load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

def init_hedron_compile_commands_repo():
    http_archive(
        name = "hedron_compile_commands",
        strip_prefix = "bazel-compile-commands-extractor-0e990032f3c5a866e72615cf67e5ce22186dcb97",
        url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/0e990032f3c5a866e72615cf67e5ce22186dcb97.tar.gz",
    )

    hedron_compile_commands_setup()
    hedron_compile_commands_setup_transitive()
    hedron_compile_commands_setup_transitive_transitive()
    hedron_compile_commands_setup_transitive_transitive_transitive()
