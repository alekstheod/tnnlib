load("//third_party/llvm:hermetic_toolchain_config.bzl", "hermetic_toolchain_config")

package(default_visibility = ["//visibility:public"])

hermetic_toolchain_config(
    name = "hermetic_toolchain_config",
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|local": ":cc-compiler-local",
        "k8|local": ":cc-compiler-local",
        "k8": ":cc-compiler-local",
        "x86_64|local": ":cc-compiler-local",
    },
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files = "@llvm_toolchain//:all",
    compiler_files = "@llvm_toolchain//:bin",
    dwp_files = "@llvm_toolchain//:all", 
    linker_files = "@llvm_toolchain//:all",
    objcopy_files = "@llvm_toolchain//:bin",
    strip_files = "@llvm_toolchain//:bin",
    toolchain_config = ":hermetic_toolchain_config",
    toolchain_identifier = "llvm-hermetic",
)

# toolchain(
#     name = "hermetic_llvm_toolchain",
#     exec_compatible_with = [],
#     target_compatible_with = [],
#     toolchain = ":toolchain",
#     toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
# )
