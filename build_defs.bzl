def if_rocm(if_true, if_false = []):
    return select({
        "//:using_rocm": if_true,
        "//conditions:default": if_false,
    })

def rocm_copts(srcs):
    return [("-x", "rocm") if s.endswith(".hip.cpp") else [] for s in srcs]

def rocm_library(name, copts = [], deps = [], **kwargs):
    native.cc_library(
        name = name,
        copts = copts + if_rocm(["-x", "rocm"]),
        deps = deps + if_rocm(["@config_rocm_hipcc//rocm:hip_runtime"]),
        **kwargs
    )

def cuda_library(name, copts = [], deps = [], **kwargs):
    native.cc_library(
        name = name,
        copts = copts + select({
            "//conditions:default": [],
        }),
        deps = deps + select({
            "//conditions:default": [],
        }),
        **kwargs
    )
