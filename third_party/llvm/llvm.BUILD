package(default_visibility = ["//visibility:public"])

filegroup(
    name = "bin",
    srcs = glob(["bin/**"]),
)

filegroup(
    name = "lib",
    srcs = glob(["lib/**"]),
)

filegroup(
    name = "include",
    srcs = glob(["include/**"]),
)

filegroup(
    name = "all",
    srcs = [
        ":bin",
        ":include",
        ":lib",
    ],
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|local": ":cc-compiler-local",
    },
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":all",
    compiler_files = ":bin",
    dwp_files = ":all",
    linker_files = ":all",
    objcopy_files = ":bin",
    strip_files = ":bin",
    toolchain_config = "@//third_party/llvm:hermetic_toolchain_config",
    toolchain_identifier = "llvm-hermetic",
)

toolchain(
    name = "hermetic_llvm_toolchain",
    exec_compatible_with = [],
    target_compatible_with = [],
    toolchain = ":toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
