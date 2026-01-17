#!/bin/bash

# Script to test hermetic LLVM toolchain setup

echo "Building with hermetic LLVM toolchain..."

# Test basic build
bazel build //...

echo "Testing OCR build..."

# Test OCR application build
bazel build //ocr:ocr

echo "Running tests..."

# Run tests
bazel test //...

echo "Hermetic LLVM toolchain test completed!"
