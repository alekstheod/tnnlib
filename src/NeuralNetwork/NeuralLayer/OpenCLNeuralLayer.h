#ifndef OPENCL_NEURAL_LAYER_H
#define OPENCL_NEURAL_LAYER_H

#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

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
            // clang-format off
            std::string src = "__kernel void dot_product(__global float* weights,"
                                                        "__global float* values,"
                                                        "__global float* result,"
                                                        "__const unsigned int sz){"
                                "float dot = 0.f;"
                                "unsigned int i;"
                                "unsigned int offset = get_global_id(0) * sz;"
                                "for( i = 0; i < sz; ++i )"
                                "{"
                                    "dot += weights[ offset + i ] * values[ offset + i ];"
                                "}"
                                "result[get_global_id(0)] = dot;"
                              "}";
            // clang-format on

            Program::Sources source(1, std::make_pair(src.c_str(), src.length() + 1));
            // Make program of the source code in the context
            cl::Program program = cl::Program(context, source);

            // Build program for these specific devices
            try {
                program.build(devices);
            } catch(const cl::Error& e) {
                cl_int err;
                cl::STRING_CLASS buildlog =
                 program.getBuildInfo< CL_PROGRAM_BUILD_LOG >(devices[0], &err);
                std::cerr << "Building error! Log: " << buildlog << std::endl;
                throw std::runtime_error{buildlog};
            }

            return program;
        }

        /// @brief OpenCL based neural layer. Used to improve the perormace
        /// for a larg ammount of neurons. This layer will use the openCL in
        /// order to calculate a dot product for the neuros inputs.
        template< class Internal >
        class OpenCLNeuralLayer {
          public:
            using Neuron = typename Internal::Neuron;
            using Var = typename Internal::Var;
            using Memento = typename Internal::Memento;

            template< template< class > class NewType >
            using wrap =
             OpenCLNeuralLayer< typename Internal::template wrap< NewType > >;

            template< unsigned int inputs >
            using resize =
             OpenCLNeuralLayer< typename Internal::template resize< inputs > >;

            template< typename VarType >
            using use =
             OpenCLNeuralLayer< typename Internal::template use< VarType > >;

            BOOST_STATIC_CONSTEXPR unsigned int CONST_NEURONS_NUMBER =
             Internal::CONST_NEURONS_NUMBER;
            BOOST_STATIC_CONSTEXPR unsigned int CONST_INPUTS_NUMBER =
             Internal::CONST_INPUTS_NUMBER;

          private:
            cl::Context m_context;
            Internal m_internal;
            cl::Program m_program;
            cl::Kernel m_kernel;
            std::array< float, CONST_INPUTS_NUMBER * CONST_NEURONS_NUMBER > m_weights;
            std::array< float, CONST_INPUTS_NUMBER * CONST_NEURONS_NUMBER > m_values;

          private:
            void calculate() {
                using namespace cl;
                // Create a command queue and use the first device
                const std::size_t size = m_weights.size();
                std::vector< Device > devices =
                 m_context.getInfo< CL_CONTEXT_DEVICES >();
                Buffer weights(m_context, CL_MEM_READ_ONLY, size * sizeof(float));
                Buffer values(m_context, CL_MEM_READ_ONLY, size * sizeof(float));
                Buffer product(m_context, CL_MEM_WRITE_ONLY, size * sizeof(float));

                // Set arguments to kernel
                m_kernel.setArg(0, weights);
                m_kernel.setArg(1, values);
                m_kernel.setArg(2, product);
                m_kernel.setArg(3, CONST_INPUTS_NUMBER);
                CommandQueue queue(m_context, devices[0]);

                try {
                    std::vector< float > dotProducts(CONST_NEURONS_NUMBER);
                    for(std::size_t i = 0; i < CONST_NEURONS_NUMBER; ++i) {
                        // Create memory buffers
                        for(std::size_t j = 0; j < CONST_INPUTS_NUMBER; ++j) {
                            const std::size_t index = i * CONST_INPUTS_NUMBER + j;
                            m_weights[index] = m_internal[i][j].weight;
                            m_values[index] = m_internal[i][j].value;
                        }
                    }

                    queue.enqueueWriteBuffer(weights,
                                             CL_TRUE,
                                             0,
                                             m_weights.size() * sizeof(float),
                                             m_weights.data());
                    queue.enqueueWriteBuffer(values,
                                             CL_TRUE,
                                             0,
                                             m_values.size() * sizeof(float),
                                             m_values.data());
                    for(int offset = 0; offset < CONST_NEURONS_NUMBER; ++offset) {
                        std::size_t rangeSize = CONST_INPUTS_NUMBER;
                        queue.enqueueNDRangeKernel(m_kernel,
                                                   cl::NDRange(offset),
                                                   cl::NDRange(rangeSize));
                    }

                    queue.enqueueReadBuffer(product,
                                            CL_TRUE,
                                            0,
                                            CONST_NEURONS_NUMBER * sizeof(float),
                                            dotProducts.data());
                    for(std::size_t i = 0; i < CONST_NEURONS_NUMBER; ++i) {
                        m_internal[i].calculateOutput(dotProducts.begin(),
                                                      dotProducts.end());
                    }
                } catch(const cl::Error& e) {
                    cl_int err;
                    cl::STRING_CLASS buildlog =
                     m_program.getBuildInfo< CL_PROGRAM_BUILD_LOG >(devices[0], &err);
                    std::cout << "Building error! Log: " << buildlog << std::endl;
                }
            }


          public:
            OpenCLNeuralLayer()
             : m_context(createContext()), m_program(createProgram(m_context)),
               m_kernel(m_program, "dot_product") {
            }

            /**
             * Constructor will initialize the layer by the given inputs number
             * and neurons number.
             */
            static_assert(CONST_NEURONS_NUMBER > 0,
                          "Invalid template argument neuronsNumber == 0");
            static_assert(CONST_INPUTS_NUMBER > 0,
                          "Invalid template argument inputsNumber <= 1");

            /**
             * @see {INeuralLayer}
             */
            auto cbegin() const {
                return m_internal.cbegin();
            }

            /**
             * @see {INeuralLayer}
             */
            auto cend() const {
                return m_internal.cend();
            }

            /**
             * @see {INeuralLayer}
             */
            auto begin() {
                return m_internal.begin();
            }

            /**
             * @see {INeuralLayer}
             */
            auto end() {
                return m_internal.end();
            }

            /**
             * @see {INeuralLayer}
             */
            unsigned int size() const {
                return m_internal.size();
            }

            /**
             * @see {INeuralLayer}
             */
            const Neuron& operator[](unsigned int id) const {
                return m_internal[id];
            }

            /**
             * @see {INeuralLayer}
             */
            void setInput(unsigned int inputId, const Var& value) {
                m_internal.setInput(inputId, value);
            }

            const Var& getBias(unsigned int neuronId) const {
                return m_internal.getBias(neuronId);
            }

            /**
             * @see {INeuralLayer}
             */
            const Var& getInputWeight(unsigned int neuronId, unsigned int weightId) const {
                return m_internal.getInputWeight(neuronId, weightId);
            }

            /**
             * @see {INeuralLayer}
             */
            const Memento getMemento() const {
                return m_internal.getMemento();
            }

            /**
             * @see {INeuralLayer}
             */
            void setMemento(const Memento& memento) {
                m_internal.setMemento(memento);
            }

            /**
             * @see {INeuralLayer}
             */
            Var getOutput(unsigned int outputId) const {
                return m_internal.getOutput(outputId);
            }

            /**
             * @see {INeuralLayer}
             */
            template< typename Layer >
            void calculateOutputs(Layer& nextLayer) {
                calculate();
                for(unsigned int i = 0; i < CONST_NEURONS_NUMBER; i++) {
                    nextLayer.setInput(i, m_internal[i].getOutput());
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
    /// @param scaleFactor a factor which will be applied during the weight
    /// initialization a final weight will be calculated in a following way
    /// random(0, 1)/scaleFactor
    template< template< template< class > class, class, std::size_t, int > class NeuronType,
              template< class > class ActivationFunctionType,
              std::size_t size,
              std::size_t inputsNumber = 2,
              int scaleFactor = 1,
              typename Var = float >
    using OpenCLNeuralLayer =
     detail::OpenCLNeuralLayer< NeuralLayer< NeuronType, ActivationFunctionType, size, inputsNumber, scaleFactor > >;
} // namespace nn

#endif
