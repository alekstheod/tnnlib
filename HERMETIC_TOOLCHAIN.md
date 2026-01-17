# Hermetic LLVM Toolchain

This PR introduces a hermetic LLVM toolchain for tnnlib, allowing the project to be compiled without requiring a system compiler.

## Changes

### 1. Hermetic LLVM Toolchain Implementation
- **`hermetic_llvm_toolchain.bzl`**: Custom Bazel rule implementing hermetic LLVM toolchain
- **`WORKSPACE`**: Added LLVM 15.0.7 download and toolchain setup
- **`BUILD.toolchain`**: Toolchain registration and implementation

### 2. Build Configuration Updates
- **`.bazelrc`**: Added `--config=hermetic` and `--config=asan_hermetic` configurations
- **`test_hermetic_toolchain.sh`**: Test script for validating hermetic toolchain

### 3. Features
- **Self-contained**: LLVM 15.0.7 toolchain downloaded and managed by Bazel
- **Cross-platform**: Works on Linux and macOS
- **Optimized**: Includes optimization flags and proper library linking
- **Debug support**: AddressSanitizer support with hermetic toolchain

## Usage

### Build with Hermetic Toolchain
```bash
bazel build --config=hermetic //...
```

### Build with AddressSanitizer
```bash
bazel build --config=asan_hermetic //...
```

### Test Hermetic Toolchain
```bash
./test_hermetic_toolchain.sh
```

## Benefits

1. **Reproducible Builds**: Eliminates dependency on system compiler variations
2. **Consistent Toolchain**: Same LLVM version across all development environments
3. **CI/CD Friendly**: No need to install compilers on build agents
4. **Simplified Setup**: New developers can build without installing compilers

## Technical Details

The hermetic toolchain provides:
- `clang` and `clang++` as C/C++ compilers
- `llvm-ar` for archiving
- `ld.lld` for linking
- `llvm-nm`, `llvm-objcopy`, `llvm-strip` for binary utilities
- Proper C++17 standard library linkage

The toolchain is automatically registered and can be used via the `--config=hermetic` flag.