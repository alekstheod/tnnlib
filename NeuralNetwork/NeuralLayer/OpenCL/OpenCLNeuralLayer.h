#pragma once

// Fixed OpenCL version compatibility with OpenCL 1.2
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>
#include <vector>

// Simple forward declarations for OpenCL types
class CLContextManager {
  private:
    cl_context context_;

  public:
    CLContextManager() : context_(nullptr) {
    }

    void initialize() {
        if(context_ != nullptr)
            return;

        cl_uint numDevices = 0;
        cl_device_id device_ids[1] = {nullptr};

        cl_int result =
         clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_DEFAULT, 1, device_ids, &numDevices);

        if(result == CL_SUCCESS && numDevices > 0) {
            context_ = clCreateContext(nullptr, 1, device_ids, nullptr, nullptr, nullptr);
        }
    }

    cl_context getContext() const {
        return context_;
    }
    bool isInitialized() const {
        return context_ != nullptr;
    }
};

// Simplified Buffer type
class CLBuffer {
  private:
    cl_mem buffer_;

  public:
    CLBuffer() : buffer_(nullptr) {
    }

    void create(cl_context context, size_t size) {
        if(!context)
            return;

        cl_int err;
        buffer_ = clCreateBuffer(context, CL_MEM_READ_WRITE, size, nullptr, &err);
    }

    cl_mem get() const {
        return buffer_;
    }
    bool isValid() const {
        return buffer_ != nullptr;
    }
};

namespace nn::detail {}