#pragma once

#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/OpenCL/OpenCLNeuralLayer.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>

#include <vector>

namespace nn {

    namespace bp {

        template< typename Internal >
        struct BPNeuralLayer< nn::detail::OpenCLNeuralLayer< Internal > >
         : private nn::detail::OpenCLNeuralLayer< Internal > {

            using Base = nn::detail::OpenCLNeuralLayer< Internal >;

            using NeuralLayerType = nn::detail::OpenCLNeuralLayer< Internal >;
            using Var = typename NeuralLayerType::Var;
            using ActivationFunctions = typename NeuralLayerType::ActivationFunctions;

          private:
            ActivationFunctions m_activationFunctions{};

          public:
            ActivationFunctions& activationFunctions() {
                return m_activationFunctions;
            }

            const ActivationFunctions& activationFunctions() const {
                return m_activationFunctions;
            }

            template< typename VarType >
            using use =
             BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

            template< std::size_t inputs >
            using adjust =
             BPNeuralLayer< typename NeuralLayerType::template adjust< inputs > >;

            using Base::calculateOutputs;
            using Base::for_each;
            using Base::getWeight;
            using Base::inputs;
            using Base::size;
            using Base::syncWeights;
            using Base::operator[];

            BPNeuralLayer() = default;

            template< typename... Args >
            BPNeuralLayer(Args&&... args)
             : Base(std::forward< Args >(args)...) {
            }

            void setInput(unsigned int inputId, const Var& value) {
                auto& self = *this;
                utils::for_< size() >([&self, inputId, &value](const auto& i) mutable {
                    const auto idx = i.value * inputs() + inputId;
                    self.m_inputs[idx] = value;
                    auto& neuron = self[i.value];
                    neuron[inputId].value = value;
                    neuron[inputId].weight = self.m_weights[idx];
                });
            }

            struct OpenCLProgram {
                cl::Context context{nn::detail::createContext()};
                std::vector< cl::Device > devices{context.getInfo< CL_CONTEXT_DEVICES >()};
                cl::Program program{
                 nn::detail::createProgram("NeuralNetwork//"
                                           "BackPropagation/OpenCL/"
                                           "calc_weights.cl",
                                           context,
                                           devices.front())};

                static OpenCLProgram& instance() {
                    static OpenCLProgram program;
                    return program;
                }
            };

            template< typename BPCtx, std::size_t myIdx >
            void calculateWeights(BPCtx& ctx, const Var& learningRate) {
                auto& deltas = std::get< myIdx >(ctx.deltas);
                auto& weights = std::get< myIdx >(ctx.weights);
                auto& biases = std::get< myIdx >(ctx.biases);
                constexpr auto inputsNum = inputs();
                constexpr auto sz = size();

                for(auto i = 0u; i < sz; ++i) {
                    for(auto j = 0u; j < inputsNum; ++j) {
                        m_weights[i * inputsNum + j] = weights[i * inputsNum + j];
                    }
                }

                std::vector< Var > deltasVec(deltas.begin(), deltas.end());

                try {
                    using namespace cl;
                    auto& ocl = OpenCLProgram::instance();
                    const auto& defaultDevice = ocl.devices.front();

                    const cl_mem_flags inBufFlags =
                     CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR;

                    const cl_mem_flags outBufFlags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;

                    Buffer values(ocl.context,
                                  inBufFlags,
                                  m_inputs.size() * sizeof(float),
                                  m_inputs.data());

                    Buffer deltasBuf(ocl.context,
                                  inBufFlags,
                                  deltasVec.size() * sizeof(float),
                                  deltasVec.data());

                    Buffer oclWeights(ocl.context,
                                   outBufFlags,
                                   m_weights.size() * sizeof(float),
                                   m_weights.data());

                    CommandQueue queue(ocl.context, defaultDevice);
                    cl::Kernel kernel{ocl.program, "calc_weights"};

                    kernel.setArg(0, values);
                    kernel.setArg(1, deltasBuf);
                    kernel.setArg(2, oclWeights);
                    kernel.setArg(3, learningRate);
                    kernel.setArg(4, static_cast< unsigned int >(Internal::inputs()));

                    queue.enqueueNDRangeKernel(kernel,
                                               cl::NullRange,
                                               cl::NDRange(sz),
                                               cl::NullRange);

                    queue.enqueueReadBuffer(oclWeights,
                                            CL_TRUE,
                                            0,
                                            m_weights.size() * sizeof(float),
                                            m_weights.data());

                    for(auto i = 0u; i < sz; ++i) {
                        for(auto j = 0u; j < inputsNum; ++j) {
                            weights[i * inputsNum + j] = m_weights[i * inputsNum + j];
                        }
                    }

                    for(auto i = 0u; i < sz; ++i) {
                        biases[i] = biases[i] - learningRate * deltas[i];
                    }

                } catch(const cl::Error& e) {
                    std::cerr << "Calculation error" << e.what() << std::endl;
                }
            }

            template< typename BPCtx, std::size_t myIdx, typename Prototype, typename MomentumFunc >
            void calculateDeltas(BPCtx& ctx, const Prototype& prototype, MomentumFunc momentum) {
                auto& outputFunc = std::get< 0 >(m_activationFunctions);
                auto& outputs = std::get< myIdx >(ctx.outputs);
                auto& deltas = std::get< myIdx >(ctx.deltas);
                utils::for_< size() >([&](auto i) {
                    deltas[i.value] = momentum(deltas[i.value],
                                               outputFunc.delta(outputs[i.value],
                                                                std::get< 1 >(prototype)[i.value]));
                });
            }

          private:
            using Base::bufferSize;
            using Base::m_inputs;
            using Base::m_weights;
        };

    } // namespace bp

} // namespace nn
