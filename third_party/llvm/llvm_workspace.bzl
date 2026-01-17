load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def init_llvm_repo():
    # Hermetic LLVM toolchain
    http_archive(
        name = "llvm_toolchain",
        build_file = "//third_party/llvm:llvm.BUILD",
        sha256 = "b3b7f2801d15d50736acea3c73982994d025b01c2f035b91ae3b49d1b575732b",
        strip_prefix = "LLVM-21.1.8-Linux-X64",
        urls = ["https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/LLVM-21.1.8-Linux-X64.tar.xz"],
    )
