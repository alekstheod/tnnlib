#pragma once

#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "Utilities/MPL/Algorithm.h"

#include <range/v3/view.hpp>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <array>
#include <iostream>

namespace nn {

    namespace detail {

        cl::Context createContext();
        bool isOpenCLAvailable();
        cl::Program createProgram(const std::string& programPath,
                                  const cl::Context& context,
                                  const cl::Device& devices);

        /// @brief OpenCL based neural layer. Used to improve the perormace
        /// for a larg ammount of neurons. This layer will use the openCL in
        /// order to calculate a dot product for the neuros inputs.
        template< class Internal >
        struct OpenCLNeuralLayer : private Internal {

            OpenCLNeuralLayer() {
                initializeOpenCLBuffers();
                syncWeights();
            }

            struct OpenCLProgram {
                cl::Context context{createContext()};
                std::vector< cl::Device > devices;
                cl::Program program;

                OpenCLProgram() {
                    devices = context.getInfo< CL_CONTEXT_DEVICES >();
                    if(!devices.empty()) {
                        program = createProgram(
                         "NeuralNetwork/NeuralLayer/OpenCL/dot_product.cl",
                         context,
                         devices.front());
                    }
                }

                static OpenCLProgram& instance() {
                    static OpenCLProgram instance;
                    return instance;
                }
            };

          private:
            void initializeOpenCLBuffers() {
                auto& ocl = OpenCLProgram::instance();
                if(ocl.devices.empty()) {
                    return;
                }
                m_weightsBuffer = cl::Buffer(ocl.context,
                                             CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                             m_weights.size() * sizeof(float));
                m_inputsBuffer = cl::Buffer(ocl.context,
                                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                            m_inputs.size() * sizeof(float));
                m_outputsBuffer = cl::Buffer(ocl.context,
                                             CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                             m_dotProducts.size() * sizeof(float));
            }

            void syncWeightsToGPU() {
                if(!m_weightsBuffer()) {
                    return;
                }
                try {
                    auto& ocl = OpenCLProgram::instance();
                    cl::CommandQueue queue(ocl.context, ocl.devices.front());
                    float* weightsHost = static_cast< float* >(queue.enqueueMapBuffer(
                     m_weightsBuffer, CL_TRUE, CL_MAP_WRITE, 0, m_weights.size() * sizeof(float)));
                    std::copy(m_weights.begin(), m_weights.end(), weightsHost);
                    queue.enqueueUnmapMemObject(m_weightsBuffer, weightsHost);
                } catch(const cl::Error&) {
                }
            }

            void syncInputsToGPU() {
                if(!m_inputsBuffer()) {
                    return;
                }
                try {
                    auto& ocl = OpenCLProgram::instance();
                    cl::CommandQueue queue(ocl.context, ocl.devices.front());
                    float* inputsHost = static_cast< float* >(queue.enqueueMapBuffer(
                     m_inputsBuffer, CL_TRUE, CL_MAP_WRITE, 0, m_inputs.size() * sizeof(float)));
                    std::copy(m_inputs.begin(), m_inputs.end(), inputsHost);
                    queue.enqueueUnmapMemObject(m_inputsBuffer, inputsHost);
                } catch(const cl::Error&) {
                }
            }

            void syncOutputsFromGPU() {
                if(!m_outputsBuffer()) {
                    return;
                }
                try {
                    auto& ocl = OpenCLProgram::instance();
                    cl::CommandQueue queue(ocl.context, ocl.devices.front());
                    float* outputsHost = static_cast< float* >(queue.enqueueMapBuffer(
                     m_outputsBuffer, CL_TRUE, CL_MAP_READ, 0, m_dotProducts.size() * sizeof(float)));
                    std::copy(outputsHost,
                              outputsHost + m_dotProducts.size(),
                              m_dotProducts.begin());
                    queue.enqueueUnmapMemObject(m_outputsBuffer, outputsHost);
                } catch(const cl::Error&) {
                }
            }

          public:
            using Var = typename Internal::Var;
            using Memento = typename Internal::Memento;
            using ActivationFunctions = typename Internal::ActivationFunctions;

            template< template< class > class NewType >
            using wrap =
             OpenCLNeuralLayer< typename Internal::template wrap< NewType > >;

            template< unsigned int inputs >
            using adjust =
             OpenCLNeuralLayer< typename Internal::template adjust< inputs > >;

            template< typename VarType >
            using use =
             OpenCLNeuralLayer< typename Internal::template use< VarType > >;

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

          public:
            void setMemento(const Memento& memento) {
                Internal::setMemento(memento);
                syncWeights();
            }

            const Var& getWeight(std::size_t neuronId, std::size_t inputId) const {
                return m_weights[neuronId * inputs() + inputId];
            }

            void setInput(unsigned int inputId, const Var& value) {
                auto& self = *this;
                utils::for_< size() >([&self, inputId, &value](const auto& i) mutable {
                    const auto idx = i.value * inputs() + inputId;
                    self.m_inputs[idx] = value;
                    self[i.value][inputId].value = value;
                    self.m_weights[idx] = self[i.value][inputId].weight;
                });
            }

            template< typename Layer >
            void calculateOutputs(Layer& nextLayer) {
                calculate();
                auto& self = *this;
                for(unsigned int i = 0; i < size(); i++) {
                    nextLayer.setInput(i, self[i].getOutput());
                }
            }

            void calculateOutputs() {
                calculate();
            }

          private:
            void calculate() {
                try {
                    using namespace cl;
                    auto& ocl = OpenCLProgram::instance();

                    if(ocl.devices.empty()) {
                        throw std::runtime_error("No OpenCL devices available");
                    }

                    const auto& defaultDevice = ocl.devices.front();

                    cl::CommandQueue queue(ocl.context, defaultDevice);

                    if(m_weightsBuffer() && m_inputsBuffer() && m_outputsBuffer()) {
                        syncWeightsToGPU();
                        syncInputsToGPU();

                        cl::Kernel kernel{ocl.program, "dot_product"};
                        kernel.setArg(0, m_weightsBuffer);
                        kernel.setArg(1, m_inputsBuffer);
                        kernel.setArg(2, m_outputsBuffer);
                        cl_uint inputs_count =
                         static_cast< cl_uint >(Internal::inputs());
                        kernel.setArg(3, sizeof(cl_uint), &inputs_count);

                        queue.enqueueNDRangeKernel(kernel,
                                                   cl::NullRange,
                                                   cl::NDRange(size()),
                                                   cl::NullRange);

                        syncOutputsFromGPU();
                    } else {
                        const cl_mem_flags inBufFlags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
                        const cl_mem_flags outBufFlags = CL_MEM_WRITE_ONLY;

                        Buffer weights(ocl.context,
                                       inBufFlags,
                                       m_weights.size() * sizeof(float),
                                       m_weights.data());

                        Buffer values(ocl.context,
                                      inBufFlags,
                                      m_inputs.size() * sizeof(float),
                                      m_inputs.data());

                        Buffer product(ocl.context,
                                       outBufFlags,
                                       m_dotProducts.size() * sizeof(float));

                        cl::Kernel kernel{ocl.program, "dot_product"};
                        kernel.setArg(0, sizeof(cl_mem), &weights);
                        kernel.setArg(1, sizeof(cl_mem), &values);
                        kernel.setArg(2, sizeof(cl_mem), &product);
                        cl_uint inputs_count =
                         static_cast< cl_uint >(Internal::inputs());
                        kernel.setArg(3, sizeof(cl_uint), &inputs_count);

                        queue.enqueueNDRangeKernel(kernel,
                                                   cl::NullRange,
                                                   cl::NDRange(size()),
                                                   cl::NullRange);

                        queue.enqueueReadBuffer(product,
                                                CL_TRUE,
                                                0,
                                                m_dotProducts.size() * sizeof(float),
                                                m_dotProducts.data());
                    }

                    // Finalize computation with bias and activation
                    auto& self = *this;
                    for(const auto i : ranges::views::indices(size())) {
                        m_dotProducts[i] += self[i].getBias();
                    }

                    for(const auto i : ranges::views::indices(size())) {
                        auto& neuron = self[i];
                        neuron.calculateOutput(m_dotProducts[i],
                                               m_dotProducts.begin(),
                                               m_dotProducts.end());
                    }
                } catch(const cl::Error& e) {
                    std::cout << "OpenCL error (" << e.err()
                              << "): " << e.what() << std::endl;
                } catch(const std::exception& e) {
                    std::cout << "Calculation error: " << e.what() << std::endl;
                }
            }

          private:
            void syncWeights() {
                auto& self = *this;
                for(const auto i : ranges::views::indices(size())) {
                    for(const auto j : ranges::views::indices(inputs())) {
                        m_weights[i * inputs() + j] = self[i][j].weight;
                    }
                }
                syncWeightsToGPU();
            }

          protected:
            static constexpr auto bufferSize = size() * inputs();
            std::array< float, bufferSize > m_weights;
            std::array< float, bufferSize > m_inputs;
            std::array< float, size() > m_dotProducts;
            cl::Buffer m_weightsBuffer;
            cl::Buffer m_inputsBuffer;
            cl::Buffer m_outputsBuffer;
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
              template< class > class ActivationFunctionType,
              std::size_t size,
              std::size_t inputsNumber = 2,
              typename Var = float >
    using OpenCLNeuralLayer =
     detail::OpenCLNeuralLayer< NeuralLayer< NeuronType, ActivationFunctionType, size, inputsNumber > >;
} // namespace nn
