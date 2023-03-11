licenses(["notice"])  #BSD

cc_library(
    name = "libpng",
    srcs = [
        "png.c",
        "pngerror.c",
        "pngget.c",
        "pngmem.c",
        "pngpread.c",
        "pngread.c",
        "pngrio.c",
        "pngrtran.c",
        "pngrutil.c",
        "pngset.c",
        "pngtrans.c",
        "pngwio.c",
        "pngwrite.c",
        "pngwtran.c",
        "pngwutil.c",
    ],
    hdrs = [
        "png.h",
        "pngconf.h",
        "pngdebug.h",
        "pnginfo.h",
        "pngpriv.h",
        "pngstruct.h",
    ],
    copts = [
        "-w",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@libpng_config",
        "@zlib",
    ],
)
