def _opencl_kernel_compile_impl(ctx):
    src = ctx.file.src
    out = ctx.outputs.out

    clang_path = "/usr/bin/clang"
    llvm_spirv_path = "/usr/bin/llvm-spirv"

    bc_file = ctx.actions.declare_file(src.basename.replace(".cl", ".bc"))
    
    ctx.actions.run(
        executable = clang_path,
        arguments = [
            "-O3",
            "-cl-std=CL3.0",
            "-target", "spirv64",
            "-emit-llvm",
            "-c",
            "-o", bc_file.path,
            "-I", "/usr/lib/clang/18/include",
            src.path,
        ],
        inputs = [src],
        outputs = [bc_file],
        progress_message = "Compiling OpenCL kernel to bitcode: %s" % src.path,
    )

    ctx.actions.run(
        executable = llvm_spirv_path,
        arguments = [
            "-o", out.path,
            bc_file.path,
        ],
        inputs = [bc_file],
        outputs = [out],
        progress_message = "Converting to SPIR-V: %s" % src.path,
    )

opencl_kernel = rule(
    implementation = _opencl_kernel_compile_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = [".cl"]),
    },
    outputs = {
        "out": "%{name}.spv",
    },
)

def _opencl_kernel_binary_impl(ctx):
    src = ctx.file.src
    out = ctx.outputs.out

    # Generate C header with binary data
    ctx.actions.run(
        executable = ctx.executable._gen_kernel_binary,
        arguments = [
            "--input", src.path,
            "--output", out.path,
            "--var_name", ctx.attr.var_name,
        ],
        inputs = [src],
        outputs = [out],
        progress_message = "Generating kernel binary header: %s" % src.path,
    )

opencl_kernel_binary = rule(
    implementation = _opencl_kernel_binary_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = [".spv"]),
        "var_name": attr.string(mandatory = True),
        "_gen_kernel_binary": attr.label(
            default = "//tools:gen_kernel_binary",
            executable = True,
            cfg = "host",
        ),
    },
    outputs = {
        "out": "%{name}.h",
    },
)
