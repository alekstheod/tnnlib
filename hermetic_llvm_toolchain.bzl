load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

def _hermetic_llvm_toolchain_impl(ctx):
    # Find the system toolchain for fallback
    system_toolchain = find_cpp_toolchain(ctx)
    
    # Create LLVM compiler paths
    llvm_bin = ctx.path(Label("@llvm_toolchain//:bin"))
    cc = llvm_bin.get_child("clang")
    cxx = llvm_bin.get_child("clang++")
    ar = llvm_bin.get_child("llvm-ar")
    nm = llvm_bin.get_child("llvm-nm")
    ld = llvm_bin.get_child("ld.lld")
    objcopy = llvm_bin.get_child("llvm-objcopy")
    strip = llvm_bin.get_child("llvm-strip")

    # Create compiler information
    cc_info = cc_common.create_cc_info(
        target_name = ctx.label.name,
        toolchain_identifier = "hermetic-llvm-x86_64",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "x86_64",
        target_libc = "unknown",
        compile_flags = [
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I{}".format(ctx.path(Label("@llvm_toolchain//:include"))),
        ],
        cxx_flags = [
            "-std=c++17",
            "-O2", 
            "-Wall",
            "-Wextra",
            "-I{}".format(ctx.path(Label("@llvm_toolchain//:include"))),
        ],
        linker_flags = [
            "-L{}".format(ctx.path(Label("@llvm_toolchain//:lib"))),
            "-lc++",
            "-lc++abi",
            "-lm",
            "-lpthread",
            "-ldl",
        ],
        compiler = str(cc),
        linker = str(ld),
        ar = str(ar),
        nm = str(nm),
        objcopy = str(objcopy),
        strip = str(strip),
        coverage_compile_flags = [],
        coverage_link_flags = [],
        includes = [
            ctx.path(Label("@llvm_toolchain//:include")),
        ],
        quote_include_paths = [],
        framework_include_paths = [],
        system_include_paths = [
            ctx.path(Label("@llvm_toolchain//:include")),
        ],
        preprocessor_defines = [],
        runtime_library_search_directories = [
            ctx.path(Label("@llvm_toolchain//:lib")),
        ],
    )

    toolchain_info = platform_common.ToolchainInfo(
        cc = cc_info,
    )

    return [
        DefaultInfo(
            files = depset([
                cc, cxx, ar, nm, ld, objcopy, strip
            ]),
        ),
        toolchain_info,
        OutputGroupInfo(
            compilation_outputs = depset(),
            linking_outputs = depset(),
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