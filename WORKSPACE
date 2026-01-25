load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Local dependencies
local_repository(
    name = "libpng_config",
    path = "third_party/libpng_config",
)

# Custom external dependencies (not available in BCR)
http_archive(
    name = "OcrSamples",
    build_file = "//third_party/samples:BUILD",
    sha256 = "165a7e4e56a2307cb74e212c83456627f5ab91516a35c51114626f488b96da77",
    urls = ["https://github.com/alekstheod/tnnlib/raw/master/ocr/samples.zip"],
)

# OpenCL headers (hermetic, always available)
http_archive(
    name = "opencl_headers",
    build_file = "//third_party/opencl:BUILD.opencl_headers",
    sha256 = "159f2a550592bae49859fee83d372acd152328fdf95c0dcd8b9409f8fad5db93",
    strip_prefix = "OpenCL-Headers-2024.10.24",
    urls = ["https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2024.10.24.tar.gz"],
)

# OpenCL ICD Loader (provides libOpenCL.so symbols)
http_archive(
    name = "opencl_icd_loader",
    build_file = "//third_party/opencl:BUILD.opencl_icd_loader",
    sha256 = "95f2f0cda375b13d2760290df044ebea9c6ff954a7d7faa0867422442c9174dc",
    strip_prefix = "OpenCL-ICD-Loader-2024.10.24",
    urls = ["https://github.com/KhronosGroup/OpenCL-ICD-Loader/archive/refs/tags/v2024.10.24.tar.gz"],
)
