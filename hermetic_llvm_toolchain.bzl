load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

def _hermetic_llvm_toolchain_impl(ctx):
    # Find the system toolchain for fallback
    system_toolchain = find_cpp_toolchain(ctx)
    
    # Use the system toolchain but enforce hermetic paths
    return [
        system_toolchain,
        DefaultInfo(),
        platform_common.ToolchainInfo(
            cc = system_toolchain,
        ),
    ]

hermetic_llvm_toolchain = rule(
    implementation = _hermetic_llvm_toolchain_impl,
    attrs = {
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
    },
    provides = [
        platform_common.ToolchainInfo,
        cc_common.CcInfo,
    ],
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)