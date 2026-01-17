#!/bin/bash

# Script to test hermetic LLVM toolchain setup

echo "Building with hermetic LLVM toolchain..."

# Test basic build with hermetic toolchain
bazel build --config=hermetic //...

echo "Testing OCR build with hermetic toolchain..."

# Test OCR application build
bazel build --config=hermetic //ocr:ocr

echo "Running tests with hermetic toolchain..."

# Run tests with hermetic toolchain
bazel test --config=hermetic //...

echo "Hermetic LLVM toolchain test completed!"
