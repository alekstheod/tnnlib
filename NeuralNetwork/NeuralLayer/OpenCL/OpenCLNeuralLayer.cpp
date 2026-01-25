#include "NeuralNetwork/NeuralLayer/OpenCL/OpenCLNeuralLayer.h"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace nn::detail {

    cl::Context createContext() {
        using namespace cl;
        // Get available platforms
        std::vector< cl::Platform > platforms;
        cl::Platform::get(&platforms);

        if(platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        // Modern OpenCL v3 context creation with properties
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                              (cl_context_properties)platforms.front()(),
                                              0};

        // Try GPU first, then fallback to any device type
        try {
            return cl::Context(CL_DEVICE_TYPE_GPU, properties);
        } catch(const cl::Error&) {
            return cl::Context(CL_DEVICE_TYPE_DEFAULT, properties);
        }
    }

    cl::Program createProgram(const std::string& programPath,
                              const cl::Context& context,
                              const cl::Device& device) {
        using namespace cl;

        std::ifstream strm(programPath, std::ios_base::binary);
        if(!strm.is_open()) {
            throw std::runtime_error("Failed to open OpenCL kernel file: " + programPath);
        }

        using It = std::istreambuf_iterator< char >;
        std::string src{(It(strm)), It()};

        Program::Sources source{1, {src}};
        cl::Program program = cl::Program{context, source};

        // Build program with OpenCL v3 options and error handling
        cl_int build_err = CL_SUCCESS;
        try {
            build_err = program.build({device}, "-cl-std=CL3.0");
        } catch(const cl::Error& e) {
            build_err = e.err();
        }

        if(build_err != CL_SUCCESS) {
            cl_int err;
            const auto buildlog =
             program.getBuildInfo< CL_PROGRAM_BUILD_LOG >(device, &err);
            std::cout << "OpenCL build error (" << build_err
                      << "): " << buildlog << std::endl;
            throw std::runtime_error("OpenCL program build failed");
        }

        return program;
    }

} // namespace nn::detail
