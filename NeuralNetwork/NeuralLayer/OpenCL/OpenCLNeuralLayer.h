#pragma once

#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"

#include <range/v3/all.hpp>

#define CL_HPP_TARGET_OPENCL_VERSION 220
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include <array>
#include <exception>
#include <fstream>

namespace nn {

    namespace detail {

        cl::Context createContext();
        cl::Program createProgram(const std::string& programPath,
                                  const cl::Context& context,
                                  const cl::Device& devices);

        /// @brief OpenCL based neural layer. Used to improve the perormace
        /// for a larg ammount of neurons. This layer will use the openCL in
        /// order to calculate a dot product for the neuros inputs.
        template< class Internal >
        struct OpenCLNeuralLayer : private Internal {

            OpenCLNeuralLayer() {
                syncWeights();
            }

            struct OpenCLProgram {
                cl::Context context{createContext()};
                std::vector< cl::Device > devices{context.getInfo< CL_CONTEXT_DEVICES >()};
                cl::Program program{
                 createProgram("NeuralNetwork/NeuralLayer/OpenCL/"
                               "dot_product.cl",
                               context,
                               devices.front())};
                static OpenCLProgram& instance() {
                    static OpenCLProgram program;
                    return program;
                }
            };

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
                    const auto& defaultDevice = ocl.devices.front();

                    // Create a command queue and use the first device
                    const cl_mem_flags inBufFlags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
                    const cl_mem_flags outBufFlags = CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR;

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
                                   m_dotProducts.size() * sizeof(float),
                                   m_dotProducts.data());

                    CommandQueue queue(ocl.context, defaultDevice);
                    cl::Kernel kernel{ocl.program, "dot_product"};

                    // Set arguments to kernel
                    kernel.setArg(0, weights);
                    kernel.setArg(1, values);
                    kernel.setArg(2, product);
                    kernel.setArg(3, static_cast< unsigned int >(Internal::inputs()));

                    queue.enqueueNDRangeKernel(kernel,
                                               cl::NullRange,
                                               cl::NDRange(size()),
                                               cl::NullRange);

                    queue.enqueueReadBuffer(product,
                                            CL_TRUE,
                                            0,
                                            m_dotProducts.size() * sizeof(float),
                                            m_dotProducts.data());

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
                    std::cerr << "Calculation error" << std::endl;
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
            }

          protected:
            static constexpr auto bufferSize = size() * inputs();
            std::array< float, bufferSize > m_weights;
            std::array< float, bufferSize > m_inputs;
            std::array< float, size() > m_dotProducts;
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
