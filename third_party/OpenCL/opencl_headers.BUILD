package(default_visibility = ["//visibility:public"])

# Header-only OpenCL library
cc_library(
    name = "OpenCL",
    hdrs = glob(["**/*.h", "**/*.hpp"]),
    includes = ["."],
    # Define OpenCL 1.2 to match what C++ wrapper expects
    copts = ["-DCL_TARGET_OPENCL_VERSION=120"],
)