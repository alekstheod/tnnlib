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
    sha256 = "407d5e109a70ec1b6cd3380ce357c21e3d3651a91caae6d0d8e1719c69a1791d",
    strip_prefix = "OpenCL-Headers-2023.12.14",
    urls = ["https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2023.12.14.tar.gz"],
)

# OpenCL ICD Loader (provides libOpenCL.so symbols)
http_archive(
    name = "opencl_icd_loader",
    build_file = "//third_party/opencl:BUILD.opencl_icd_loader",
    sha256 = "af8df96f1e1030329e8d4892ba3aa761b923838d4c689ef52d97822ab0bd8917",
    strip_prefix = "OpenCL-ICD-Loader-2023.12.14",
    urls = ["https://github.com/KhronosGroup/OpenCL-ICD-Loader/archive/refs/tags/v2023.12.14.tar.gz"],
)
