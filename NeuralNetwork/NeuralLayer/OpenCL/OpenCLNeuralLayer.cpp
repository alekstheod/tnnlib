#include "NeuralNetwork/NeuralLayer/OpenCL/OpenCLNeuralLayer.h"

namespace nn {
    namespace detail {

        cl::Context createContext() {
            using namespace cl;
            // Get available platforms
            std::vector< cl::Platform > platforms;
            cl::Platform::get(&platforms);

            // Select the default platform and create a context using this
            // platform and the GPU
            cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
                                            (cl_context_properties)(platforms.front())(),
                                            0};

            return cl::Context(CL_DEVICE_TYPE_DEFAULT, cps);
        }

        cl::Program createProgram(const std::string& programPath,
                                  const cl::Context& context,
                                  const cl::Device& device) {
            using namespace cl;

            std::ifstream strm(programPath, std::ios_base::binary);

            using It = std::istreambuf_iterator< char >;
            std::string src{(It(strm)), It()};

            Program::Sources source{1, {src}};
            cl::Program program = cl::Program{context, source};

            // Build program for these specific devices
            try {
                program.build({device});
            } catch(const cl::Error& e) {
                cl_int err;
                const auto buildlog =
                 program.getBuildInfo< CL_PROGRAM_BUILD_LOG >(device, &err);
                std::cerr << "Building error! Log: " << buildlog << std::endl;
                throw std::runtime_error{"Build opencl program error"};
            }

            return program;
        }

    } // namespace detail

} // namespace nn
