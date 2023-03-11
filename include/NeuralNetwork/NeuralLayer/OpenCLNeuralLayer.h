#pragma once

#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>

#include <range/v3/all.hpp>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include <array>
#include <exception>
#include <fstream>

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
                                            (cl_context_properties)(platforms[0])(),
                                            0};

            return cl::Context(CL_DEVICE_TYPE_GPU, cps);
        }

        cl::Program createProgram(const cl::Context& context) {
            using namespace cl;
            // Get a list of devices on this platform
            std::vector< Device > devices = context.getInfo< CL_CONTEXT_DEVICES >();
            std::string src =
             "__kernel void dot_product(__global float* weights,         \
                                        __global float* values,          \
                                        __global float* result,          \
                                        __const unsigned int sz){        \
                                float dot = 0.f;                                         \
                                unsigned int i;                                          \
                                unsigned int idx = get_global_id(0);                     \
                                unsigned int offset = idx * sz;                          \
                                for( i = 0; i < sz; ++i )                                \
                                {                                                        \
                                    dot += weights[ offset + i ] * values[ offset + i ]; \
                                }                                                        \
                                result[idx] = dot;                                       \
                              }";

            Program::Sources source{1, {src}};
            cl::Program program = cl::Program{context, source};

            // Build program for these specific devices
            try {
                program.build(devices);
            } catch(const cl::Error& e) {
                cl_int err;
                const auto buildlog =
                 program.getBuildInfo< CL_PROGRAM_BUILD_LOG >(devices[0], &err);
                std::cerr << "Building error! Log: " << buildlog << std::endl;
                throw std::runtime_error{"Build opencl program error"};
            }

            return program;
        }

        /// @brief OpenCL based neural layer. Used to improve the perormace
        /// for a larg ammount of neurons. This layer will use the openCL in
        /// order to calculate a dot product for the neuros inputs.
        template< class Internal >
        class OpenCLNeuralLayer : Internal {
          public:
            using Var = typename Internal::Var;
            using Memento = typename Internal::Memento;

            template< template< class > class NewType >
            using wrap =
             OpenCLNeuralLayer< typename Internal::template wrap< NewType > >;

            template< unsigned int inputs >
            using adjust =
             OpenCLNeuralLayer< typename Internal::template adjust< inputs > >;

            template< typename VarType >
            using use =
             OpenCLNeuralLayer< typename Internal::template use< VarType > >;

            BOOST_STATIC_CONSTEXPR unsigned int CONST_NEURONS_NUMBER = Internal::size();
            BOOST_STATIC_CONSTEXPR unsigned int CONST_INPUTS_NUMBER =
             Internal::CONST_INPUTS_NUMBER;

          private:
            cl::Context m_context;
            cl::Program m_program;
            cl::Kernel m_kernel;
            std::vector< cl::Device > m_devices;

          private:
            void calculate() {
                using namespace cl;
                constexpr auto size = CONST_INPUTS_NUMBER * CONST_NEURONS_NUMBER;
                std::array< float, size > in_weights;
                std::array< float, size > in_values;
                // Create a command queue and use the first device
                Buffer weights(m_context, CL_MEM_READ_ONLY, size * sizeof(float));
                Buffer values(m_context, CL_MEM_READ_ONLY, size * sizeof(float));
                Buffer product(m_context, CL_MEM_WRITE_ONLY, CONST_NEURONS_NUMBER * sizeof(float));

                // Set arguments to kernel
                m_kernel.setArg(0, weights);
                m_kernel.setArg(1, values);
                m_kernel.setArg(2, product);
                m_kernel.setArg(3, CONST_INPUTS_NUMBER);
                CommandQueue queue(m_context, m_devices[0]);

                try {
                    for(const auto i : ranges::views::indices(CONST_NEURONS_NUMBER)) {
                        for(const auto j : ranges::views::indices(CONST_INPUTS_NUMBER)) {
                            const std::size_t idx = i * CONST_INPUTS_NUMBER + j;
                            in_weights[idx] = operator[](i)[j].weight;
                            in_values[idx] = operator[](i)[j].value;
                        }
                    }

                    queue.enqueueWriteBuffer(weights,
                                             CL_TRUE,
                                             0,
                                             in_weights.size() * sizeof(float),
                                             in_weights.data());

                    queue.enqueueWriteBuffer(values,
                                             CL_TRUE,
                                             0,
                                             in_values.size() * sizeof(float),
                                             in_values.data());

                    queue.enqueueNDRangeKernel(m_kernel,
                                               cl::NullRange,
                                               cl::NDRange(CONST_NEURONS_NUMBER));

                    std::array< float, CONST_NEURONS_NUMBER > dotProducts;
                    queue.enqueueReadBuffer(product,
                                            CL_TRUE,
                                            0,
                                            CONST_NEURONS_NUMBER * sizeof(float),
                                            dotProducts.data());

                    for(const auto i : ranges::views::indices(CONST_NEURONS_NUMBER)) {
                        dotProducts[i] += operator[](i).getBias();
                    }

                    for(const auto i : ranges::views::indices(CONST_NEURONS_NUMBER)) {
                        operator[](i).calculateOutput(dotProducts.begin(),
                                                      dotProducts.end());
                    }
                } catch(const cl::Error& e) {
                    std::cerr << "Calculation error" << std::endl;
                }
            }

          public:
            OpenCLNeuralLayer()
             : m_context(createContext()), m_program(createProgram(m_context)),
               m_kernel(m_program, "dot_product"),
               m_devices(m_context.getInfo< CL_CONTEXT_DEVICES >()) {
            }

            static_assert(CONST_NEURONS_NUMBER > 0,
                          "Invalid template argument neuronsNumber == 0");
            static_assert(CONST_INPUTS_NUMBER > 0,
                          "Invalid template argument inputsNumber <= 1");
            static_assert(std::is_same< float, Var >::value,
                          "VarType must be float");

            using Internal::begin;
            using Internal::cbegin;
            using Internal::cend;
            using Internal::end;
            using Internal::size;
            using Internal::operator[];
            using Internal::for_each;
            using Internal::getMemento;
            using Internal::getOutput;
            using Internal::inputs;
            using Internal::setInput;
            using Internal::setMemento;

            /**
             * @see {INeuralLayer}
             */
            template< typename Layer >
            void calculateOutputs(Layer& nextLayer) {
                calculate();
                for(unsigned int i = 0; i < CONST_NEURONS_NUMBER; i++) {
                    nextLayer.setInput(i, operator[](i).getOutput());
                }
            }

            /**
             * @see {INeuralLayer}
             */
            void calculateOutputs() {
                calculate();
            }
        };
    } // namespace detail

    /// @brief OpenCL based neural layer @see={detail::OpenCLNeuralLayer}
    /// @param NeuronType a type of the neuron in a layer.
    /// @param ActivationFunction a type of the activation function used in a
    /// neuron.
    /// @param size ammount of neurons in a layer.
    /// @param inputsNumber the number of inputs of each neuron in a layer.
    /// initialization a final weight will be calculated in a following way
    /// random(0, 1)/scaleFactor
    template< template< template< class > class, class, std::size_t > class NeuronType,
              template< class >
              class ActivationFunctionType,
              std::size_t size,
              std::size_t inputsNumber = 2,
              typename Var = float >
    using OpenCLNeuralLayer =
     detail::OpenCLNeuralLayer< NeuralLayer< NeuronType, ActivationFunctionType, size, inputsNumber > >;
} // namespace nn
