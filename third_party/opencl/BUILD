package(default_visibility = ["//visibility:public"])

# Combined OpenCL library with headers and ICD loader
cc_library(
    name = "opencl",
    copts = ["-DCL_TARGET_OPENCL_VERSION=300"],
    deps = [
        "@opencl_headers//:opencl_headers",
        "@opencl_icd_loader//:opencl_icd_loader",
    ],
)