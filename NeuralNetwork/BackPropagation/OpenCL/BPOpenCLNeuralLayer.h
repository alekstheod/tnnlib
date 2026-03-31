#pragma once

#include "NeuralNetwork/NeuralLayer/OpenCL/OpenCLNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/BPNeuralLayer.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>

#include <array>
#include <vector>

namespace nn {

    namespace bp {

        template< typename Internal >
        struct BPNeuralLayer< nn::detail::OpenCLNeuralLayer< Internal > >
         : public BPNeuralLayer< Internal > {

            using Base = BPNeuralLayer< Internal >;

            using NeuralLayerType = nn::detail::OpenCLNeuralLayer< Internal >;
            using Var = typename NeuralLayerType::Var;

            template< typename VarType >
            using use =
             BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

            template< std::size_t inputs >
            using adjust =
             BPNeuralLayer< typename NeuralLayerType::template adjust< inputs > >;

            using Memento = typename Base::Memento;

            using Base::begin;
            using Base::cbegin;
            using Base::cend;
            using Base::end;
            using Base::for_each;
            using Base::getMemento;
            using Base::getOutput;
            using Base::inputs;
            using Base::setMemento;
            using Base::size;
            using Base::operator[];
            using Base::calculateOutputs;
            using Base::setInput;

            template< typename Prototype, typename MomentumFunc >
            void calculateDeltas(const Prototype& prototype, MomentumFunc momentum) {
                auto& outputs = std::get< 1 >(prototype);
                for(std::size_t neuronId = 0; neuronId < size(); ++neuronId) {
                    auto& neuron = (*this)[neuronId];
                    auto expectedOutput = outputs[neuronId];
                    auto actualOutput = neuron.getOutput();
                    auto delta =
                     momentum(m_deltas[neuronId],
                              neuron.getOutputFunction().delta(actualOutput, expectedOutput));
                    m_deltas[neuronId] = delta;
                }
            }

            std::vector< Var >& deltas() {
                return m_deltas;
            }

            const std::vector< Var >& deltas() const {
                return m_deltas;
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

            const Var& getWeight(std::size_t neuronId, std::size_t inputId) const {
                return m_weights[neuronId * inputs() + inputId];
            }

            void calculateWeights(const Var& learningRate) {
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

                    Buffer deltas(ocl.context,
                                  inBufFlags,
                                  m_deltas.size() * sizeof(float),
                                  m_deltas.data());

                    Buffer weights(ocl.context,
                                   outBufFlags,
                                   m_weights.size() * sizeof(float),
                                   m_weights.data());

                    CommandQueue queue(ocl.context, defaultDevice);
                    cl::Kernel kernel{ocl.program, "calc_weights"};

                    kernel.setArg(0, values);
                    kernel.setArg(1, deltas);
                    kernel.setArg(2, weights);
                    kernel.setArg(3, learningRate);
                    kernel.setArg(4, static_cast< unsigned int >(Internal::inputs()));

                    queue.enqueueNDRangeKernel(kernel,
                                               cl::NullRange,
                                               cl::NDRange(size()),
                                               cl::NullRange);

                    queue.enqueueReadBuffer(weights,
                                            CL_TRUE,
                                            0,
                                            m_weights.size() * sizeof(float),
                                            m_weights.data());

                    for_each([this, &learningRate](auto i, auto& neuron) {
                        Var weight = neuron.getBias();
                        Var newWeight = weight - learningRate * m_deltas[i.value];
                        neuron.setBias(newWeight);
                    });

                } catch(const cl::Error& e) {
                    std::cerr << "Calculation error" << e.what() << std::endl;
                }
            }

            const Var& getDelta(std::size_t neuronId) const {
                return m_deltas[neuronId];
            }

            BPNeuralLayer() : m_deltas(size(), Var{}) {
            }

          private:
            static constexpr auto bufferSize = size() * inputs();
            std::array< float, bufferSize > m_weights;
            std::array< float, bufferSize > m_inputs;
            std::vector< Var > m_deltas;
        };

    } // namespace bp

} // namespace nn