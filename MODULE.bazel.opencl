# OpenCL dependency (optional)
opencl = use_extension("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
opencl.archive(
    name = "opencl_headers",
    build_file = "//third_party/OpenCL:opencl_headers.BUILD",
    sha256 = "8c2b87755a0828e9a0ba5a3a4910a73d8d57f6b59678dca7aa5b808e7d5d95c",
    urls = ["https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2023.12.14.tar.gz"],
    strip_prefix = "OpenCL-Headers-v2023.12.14",
)
use_repo(opencl, "opencl_headers")