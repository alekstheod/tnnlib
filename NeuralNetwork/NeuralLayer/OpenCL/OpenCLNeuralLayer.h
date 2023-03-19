#pragma once

#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"

#include <range/v3/all.hpp>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include <array>
#include <exception>
#include <fstream>

namespace nn {

    namespace detail {

        cl::Context createContext();
        cl::Program createProgram(const cl::Context& context,
                                  const std::vector< cl::Device >& devices);

        struct OpenCLProgram {
            cl::Context context{createContext()};
            std::vector< cl::Device > devices{context.getInfo< CL_CONTEXT_DEVICES >()};
            cl::Program program{createProgram(context, devices)};
            cl::Kernel kernel{program, "dot_product"};
            static OpenCLProgram& instance() {
                static OpenCLProgram program;
                return program;
            }
        };

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

            BOOST_STATIC_CONSTEXPR unsigned int CONST_INPUTS_NUMBER =
             Internal::CONST_INPUTS_NUMBER;

          private:
            void calculate() {
                using namespace cl;
                constexpr auto bufferSize = size() * CONST_INPUTS_NUMBER;
                std::array< float, bufferSize > in_weights;
                std::array< float, bufferSize > in_values;

                auto& ocl = OpenCLProgram::instance();

                // Create a command queue and use the first device
                Buffer weights(ocl.context, CL_MEM_READ_ONLY, bufferSize * sizeof(float));
                Buffer values(ocl.context, CL_MEM_READ_ONLY, bufferSize * sizeof(float));
                Buffer product(ocl.context, CL_MEM_WRITE_ONLY, size() * sizeof(float));

                CommandQueue queue(ocl.context, ocl.devices[0]);

                try {
                    for(const auto i : ranges::views::indices(size())) {
                        for(const auto j : ranges::views::indices(CONST_INPUTS_NUMBER)) {
                            const std::size_t idx = i * CONST_INPUTS_NUMBER + j;
                            in_weights[idx] = operator[](i)[j].weight;
                            in_values[idx] = operator[](i)[j].value;
                        }
                    }

                    // Set arguments to kernel
                    ocl.kernel.setArg(0, weights);
                    ocl.kernel.setArg(1, values);
                    ocl.kernel.setArg(2, product);
                    ocl.kernel.setArg(3, CONST_INPUTS_NUMBER);

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

                    queue.enqueueNDRangeKernel(ocl.kernel, cl::NullRange, cl::NDRange(size()));

                    std::array< float, size() > dotProducts;
                    queue.enqueueReadBuffer(
                     product, CL_TRUE, 0, size() * sizeof(float), dotProducts.data());

                    for(const auto i : ranges::views::indices(size())) {
                        dotProducts[i] += operator[](i).getBias();
                    }

                    for(const auto i : ranges::views::indices(size())) {
                        auto& neuron = operator[](i);
                        neuron.calculateOutput(dotProducts[i],
                                               dotProducts.begin(),
                                               dotProducts.end());
                    }
                } catch(const cl::Error& e) {
                    std::cerr << "Calculation error" << std::endl;
                }
            }

          public:
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

            template< typename Layer >
            void calculateOutputs(Layer& nextLayer) {
                calculate();
                for(unsigned int i = 0; i < size(); i++) {
                    nextLayer.setInput(i, operator[](i).getOutput());
                }
            }

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
