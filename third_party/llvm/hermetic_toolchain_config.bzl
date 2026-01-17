# Hermetic LLVM toolchain configuration for Bazel 7.x
# Using the correct API with individual imports

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "artifact_name_pattern", 
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
)

def _hermetic_toolchain_config_impl(ctx):
    # Define tool paths for LLVM
    tool_paths = [
        tool_path(name = "ar", path = "external/llvm_toolchain/bin/llvm-ar"),
        tool_path(name = "cpp", path = "external/llvm_toolchain/bin/clang++"),
        tool_path(name = "gcc", path = "external/llvm_toolchain/bin/clang"),
        tool_path(name = "clang", path = "external/llvm_toolchain/bin/clang"),
        tool_path(name = "clang++", path = "external/llvm_toolchain/bin/clang++"),
        tool_path(name = "ld", path = "external/llvm_toolchain/bin/ld.lld"),
        tool_path(name = "nm", path = "external/llvm_toolchain/bin/llvm-nm"),
        tool_path(name = "objcopy", path = "external/llvm_toolchain/bin/llvm-objcopy"),
        tool_path(name = "objdump", path = "external/llvm_toolchain/bin/llvm-objdump"),
        tool_path(name = "strip", path = "external/llvm_toolchain/bin/llvm-strip"),
    ]
    
    # Define features for LLVM
    features = [
        feature(
            name = "opt",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [
                        flag_group(flags = ["-O2"]),
                    ],
                ),
            ],
        ),
        feature(
            name = "compile",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [
                        flag_group(flags = ["-std=c++17", "-Wall", "-Wextra"]),
                    ],
                ),
            ],
        ),
        feature(
            name = "link",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.cpp_link_executable],
                    flag_groups = [
                        flag_group(flags = ["-lstdc++", "-lm"]),
                    ],
                ),
            ],
        ),
    ]
    
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        toolchain_identifier = "llvm-hermetic",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "x86_64",
        target_libc = "unknown",
        abi_version = "local",
        abi_libc_version = "local",
        compiler = "clang",
        cxx_builtin_include_directories = [
            "/usr/include",
            "external/llvm_toolchain/include",
            "include",
        ],
        tool_paths = tool_paths,
    )

hermetic_toolchain_config = rule(
    implementation = _hermetic_toolchain_config_impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
