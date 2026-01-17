load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def init_ocr_samples_repo():
    http_archive(
        name = "OcrSamples",
        build_file = "//third_party/OcrSamples:BUILD",
        sha256 = "165a7e4e56a2307cb74e212c83456627f5ab91516a35c51114626f488b96da77",
        urls = ["https://github.com/alekstheod/tnnlib/raw/master/ocr/samples.zip"],
    )