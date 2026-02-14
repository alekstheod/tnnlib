package(default_visibility = ["//visibility:public"])

# Combined OpenCL library with headers and ICD loader
cc_library(
    name = "opencl",
    copts = [
        "-DCL_TARGET_OPENCL_VERSION=300",
    ],
    local_defines = ["CL_TARGET_OPENCL_VERSION=300"],
    strip_include_prefix = "/usr/include",
    deps = [
        "@opencl_headers",
        "@opencl_clhpp",
        "@opencl_icd_loader",
    ],
)
