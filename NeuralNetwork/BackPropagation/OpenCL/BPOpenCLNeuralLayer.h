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

            template< typename VarType >
            using use =
             BPNeuralLayer< typename NeuralLayerType::template use< VarType > >;

            template< std::size_t inputs >
            using adjust =
             BPNeuralLayer< typename NeuralLayerType::template adjust< inputs > >;

            using Memento = typename Base::Memento;
            using Base::calculateOutputs;
            using Base::for_each;
            using Base::getMemento;
            using Base::getOutput;
            using Base::getWeight;
            using Base::inputs;
            using Base::setMemento;
            using Base::size;
            using Base::operator[];
            using Base::m_inputsBuffer;
            using Base::syncInputsToGPU;

            BPNeuralLayer() {
                initializeBPState();
            }

            template< typename... Args >
            BPNeuralLayer(Args&&... args)
             : Base(std::forward< Args >(args)...) {
                initializeBPState();
            }

          private:
            void initializeBPState() {
                m_deltas.resize(size(), Var{});
                m_accumulatedBiasGradient.resize(size(), Var{});
                m_accumulatedWeightGradients.resize(size());
                for(std::size_t i = 0; i < size(); ++i) {
                    m_accumulatedWeightGradients[i].resize(inputs(), Var{});
                }
                initializeOpenCLBuffers();
            }

            void initializeOpenCLBuffers() {
                if(OpenCLProgram::instance().devices.empty()) {
                    return;
                }
                auto& ocl = OpenCLProgram::instance();
                m_weightsBuffer = cl::Buffer(ocl.context,
                                             CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                             m_weights.size() * sizeof(float));
                m_weightGradsBuffer = cl::Buffer(ocl.context,
                                                 CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                                 m_weights.size() * sizeof(float));
                m_biasGradsBuffer =
                 cl::Buffer(ocl.context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            m_accumulatedBiasGradient.size() * sizeof(float));
                m_deltasBuffer = cl::Buffer(ocl.context,
                                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                            m_deltas.size() * sizeof(float));
                m_outputsBuffer = cl::Buffer(ocl.context,
                                             CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                             size() * sizeof(float));
                m_expectedBuffer = cl::Buffer(ocl.context,
                                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                              size() * sizeof(float));
                syncWeightsToGPU();
                syncGradientsToGPU();
                syncDeltasToGPU();
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

            void syncGradientsToGPU() {
                if(!m_weightGradsBuffer()) {
                    return;
                }
                try {
                    auto& ocl = OpenCLProgram::instance();
                    cl::CommandQueue queue(ocl.context, ocl.devices.front());
                    float* gradsHost = static_cast< float* >(queue.enqueueMapBuffer(
                     m_weightGradsBuffer, CL_TRUE, CL_MAP_WRITE, 0, m_weights.size() * sizeof(float)));
                    std::fill(gradsHost, gradsHost + m_weights.size(), 0.0f);
                    queue.enqueueUnmapMemObject(m_weightGradsBuffer, gradsHost);

                    float* biasGradsHost = static_cast< float* >(
                     queue.enqueueMapBuffer(m_biasGradsBuffer,
                                            CL_TRUE,
                                            CL_MAP_WRITE,
                                            0,
                                            m_accumulatedBiasGradient.size() * sizeof(float)));
                    std::fill(biasGradsHost,
                              biasGradsHost + m_accumulatedBiasGradient.size(),
                              0.0f);
                    queue.enqueueUnmapMemObject(m_biasGradsBuffer, biasGradsHost);
                } catch(const cl::Error&) {
                }
            }

            void syncDeltasToGPU() {
                if(!m_deltasBuffer()) {
                    return;
                }
                try {
                    auto& ocl = OpenCLProgram::instance();
                    cl::CommandQueue queue(ocl.context, ocl.devices.front());
                    float* deltasHost = static_cast< float* >(queue.enqueueMapBuffer(
                     m_deltasBuffer, CL_TRUE, CL_MAP_WRITE, 0, m_deltas.size() * sizeof(float)));
                    std::copy(m_deltas.begin(), m_deltas.end(), deltasHost);
                    queue.enqueueUnmapMemObject(m_deltasBuffer, deltasHost);
                } catch(const cl::Error&) {
                }
            }

            void syncDeltasFromGPU() {
                if(!m_deltasBuffer()) {
                    return;
                }
                try {
                    auto& ocl = OpenCLProgram::instance();
                    cl::CommandQueue queue(ocl.context, ocl.devices.front());
                    float* deltasHost = static_cast< float* >(queue.enqueueMapBuffer(
                     m_deltasBuffer, CL_TRUE, CL_MAP_READ, 0, m_deltas.size() * sizeof(float)));
                    std::copy(deltasHost, deltasHost + m_deltas.size(), m_deltas.begin());
                    queue.enqueueUnmapMemObject(m_deltasBuffer, deltasHost);
                } catch(const cl::Error&) {
                }
            }

            void syncWeightsFromGPU() {
                if(!m_weightsBuffer()) {
                    return;
                }
                try {
                    auto& ocl = OpenCLProgram::instance();
                    cl::CommandQueue queue(ocl.context, ocl.devices.front());
                    float* weightsHost = static_cast< float* >(queue.enqueueMapBuffer(
                     m_weightsBuffer, CL_TRUE, CL_MAP_READ, 0, m_weights.size() * sizeof(float)));
                    std::copy(weightsHost,
                              weightsHost + m_weights.size(),
                              m_weights.begin());
                    queue.enqueueUnmapMemObject(m_weightsBuffer, weightsHost);
                } catch(const cl::Error&) {
                }
            }

            void syncGradientsFromGPU() {
                if(!m_weightGradsBuffer()) {
                    return;
                }
                try {
                    auto& ocl = OpenCLProgram::instance();
                    cl::CommandQueue queue(ocl.context, ocl.devices.front());
                    float* gradsHost = static_cast< float* >(queue.enqueueMapBuffer(
                     m_weightGradsBuffer, CL_TRUE, CL_MAP_READ, 0, m_weights.size() * sizeof(float)));
                    std::copy(gradsHost, gradsHost + m_weights.size(), m_weights.begin());
                    queue.enqueueUnmapMemObject(m_weightGradsBuffer, gradsHost);

                    float* biasGradsHost = static_cast< float* >(
                     queue.enqueueMapBuffer(m_biasGradsBuffer,
                                            CL_TRUE,
                                            CL_MAP_READ,
                                            0,
                                            m_accumulatedBiasGradient.size() * sizeof(float)));
                    std::copy(biasGradsHost,
                              biasGradsHost + m_accumulatedBiasGradient.size(),
                              m_accumulatedBiasGradient.begin());
                    queue.enqueueUnmapMemObject(m_biasGradsBuffer, biasGradsHost);

                    for(auto i = 0u; i < size(); ++i) {
                        for(auto j = 0u; j < inputs(); ++j) {
                            m_accumulatedWeightGradients[i][j] =
                             m_weights[i * inputs() + j];
                        }
                    }
                } catch(const cl::Error&) {
                }
            }

          public:
            void syncWeights() {
                syncWeightsToGPU();
                auto& self = *this;
                for(const auto i : ranges::views::indices(size())) {
                    for(const auto j : ranges::views::indices(inputs())) {
                        m_weights[i * inputs() + j] = self[i][j].weight;
                    }
                }
            }

          public:
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
                                           "bp_kernels.cl",
                                           context,
                                           devices.front())};

                static OpenCLProgram& instance() {
                    static OpenCLProgram program;
                    return program;
                }
            };

            void calculateWeights(const Var& learningRate) {
                if(!m_weightsBuffer()) {
                    for_each([this, &learningRate](auto i, auto& neuron) {
                        std::size_t inputsNumber = neuron.size();
                        auto delta = m_deltas[i.value];
                        for(std::size_t j = 0; j < inputsNumber; j++) {
                            auto input = neuron[j].value;
                            auto weight = neuron[j].weight;
                            auto newWeight = weight - learningRate * input * delta;
                            neuron.setWeight(j, newWeight);
                        }
                        Var weight = neuron.getBias();
                        Var newWeight = weight - learningRate * delta;
                        neuron.setBias(newWeight);
                    });
                    return;
                }
                try {
                    using namespace cl;
                    auto& ocl = OpenCLProgram::instance();
                    const auto& defaultDevice = ocl.devices.front();

                    cl::CommandQueue queue(ocl.context, defaultDevice);

                    syncInputsToGPU();
                    syncDeltasToGPU();

                    cl::Kernel kernel{ocl.program, "calc_weights"};
                    kernel.setArg(0, m_inputsBuffer);
                    kernel.setArg(1, m_deltasBuffer);
                    kernel.setArg(2, m_weightsBuffer);
                    kernel.setArg(3, learningRate);
                    kernel.setArg(4, static_cast< unsigned int >(Internal::inputs()));

                    queue.enqueueNDRangeKernel(kernel,
                                               cl::NullRange,
                                               cl::NDRange(size()),
                                               cl::NullRange);

                    syncWeightsFromGPU();

                    for(auto i = 0u; i < size(); ++i) {
                        for(auto j = 0u; j < inputs(); ++j) {
                            (*this)[i][j].weight = m_weights[i * inputs() + j];
                        }
                    }

                    for_each([this, &learningRate](auto i, auto& neuron) {
                        Var weight = neuron.getBias();
                        Var newWeight = weight - learningRate * m_deltas[i.value];
                        neuron.setBias(newWeight);
                    });

                } catch(const cl::Error& e) {
                    std::cerr << "Calculation error" << e.what() << std::endl;
                }
            }

            template< typename Prototype, typename MomentumFunc >
            void calculateDeltas(const Prototype& prototype, MomentumFunc momentum) {
                try {
                    using namespace cl;
                    auto& ocl = OpenCLProgram::instance();
                    const auto& defaultDevice = ocl.devices.front();

                    const cl_mem_flags bufFlags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;

                    auto& self = *this;
                    std::vector< float > outputs(size());
                    for(std::size_t i = 0; i < size(); ++i) {
                        outputs[i] = self[i].getOutput();
                    }
                    auto& expected = std::get< 1 >(prototype);

                    cl::CommandQueue queue(ocl.context, defaultDevice);

                    Buffer outputsBuf(ocl.context, bufFlags, outputs.size() * sizeof(float));
                    Buffer expectedBuf(ocl.context, bufFlags, expected.size() * sizeof(float));
                    Buffer deltasBuf(ocl.context, bufFlags, m_deltas.size() * sizeof(float));

                    float* outputsHost = static_cast< float* >(queue.enqueueMapBuffer(
                     outputsBuf, CL_TRUE, CL_MAP_WRITE, 0, outputs.size() * sizeof(float)));
                    float* expectedHost = static_cast< float* >(queue.enqueueMapBuffer(
                     expectedBuf, CL_TRUE, CL_MAP_WRITE, 0, expected.size() * sizeof(float)));
                    float* deltasHost = static_cast< float* >(queue.enqueueMapBuffer(
                     deltasBuf, CL_TRUE, CL_MAP_WRITE, 0, m_deltas.size() * sizeof(float)));

                    std::copy(outputs.begin(), outputs.end(), outputsHost);
                    std::copy(expected.begin(), expected.end(), expectedHost);

                    queue.enqueueUnmapMemObject(outputsBuf, outputsHost);
                    queue.enqueueUnmapMemObject(expectedBuf, expectedHost);
                    queue.enqueueUnmapMemObject(deltasBuf, deltasHost);

                    cl::Kernel kernel{ocl.program, "calc_output_deltas"};
                    kernel.setArg(0, outputsBuf);
                    kernel.setArg(1, expectedBuf);
                    kernel.setArg(2, deltasBuf);
                    float mom = momentum(Var{}, Var{});
                    kernel.setArg(3, sizeof(float), &mom);
                    cl_uint sz = static_cast< cl_uint >(size());
                    kernel.setArg(4, sizeof(cl_uint), &sz);

                    queue.enqueueNDRangeKernel(kernel,
                                               cl::NullRange,
                                               cl::NDRange(size()),
                                               cl::NullRange);

                    float* resultDeltas = static_cast< float* >(queue.enqueueMapBuffer(
                     deltasBuf, CL_TRUE, CL_MAP_READ, 0, m_deltas.size() * sizeof(float)));
                    std::copy(resultDeltas,
                              resultDeltas + m_deltas.size(),
                              m_deltas.begin());
                    queue.enqueueUnmapMemObject(deltasBuf, resultDeltas);
                } catch(const cl::Error& e) {
                    std::cerr << "OpenCL error: " << e.what() << std::endl;
                    auto& self = *this;
                    auto& outputFunc = std::get< 0 >(m_activationFunctions);
                    utils::for_< size() >([&](auto i) {
                        auto& neuron = self[i.value];
                        auto delta =
                         momentum(m_deltas[i.value],
                                  outputFunc.delta(neuron.getOutput(),
                                                   std::get< 1 >(prototype)[i.value]));
                        m_deltas[i.value] = delta;
                    });
                }
            }

            template< typename AffectedLayer, typename MomentumFunc >
            void calculateHiddenDeltas(AffectedLayer& affectedLayer, MomentumFunc momentum) {
                try {
                    using namespace cl;
                    auto& ocl = OpenCLProgram::instance();
                    const auto& defaultDevice = ocl.devices.front();

                    const cl_mem_flags bufFlags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;

                    auto& nextDeltas = affectedLayer.deltas();
                    auto& currentOutputs = m_dotProducts;

                    Buffer nextDeltasBuf(ocl.context, bufFlags, nextDeltas.size() * sizeof(float));
                    Buffer weightsBuf(ocl.context, bufFlags, m_weights.size() * sizeof(float));
                    Buffer outputsBuf(ocl.context,
                                      bufFlags,
                                      currentOutputs.size() * sizeof(float));
                    Buffer deltasBuf(ocl.context, bufFlags, m_deltas.size() * sizeof(float));

                    cl::CommandQueue queue(ocl.context, defaultDevice);

                    float* nextDeltasHost = static_cast< float* >(queue.enqueueMapBuffer(
                     nextDeltasBuf, CL_TRUE, CL_MAP_WRITE, 0, nextDeltas.size() * sizeof(float)));
                    float* weightsHost = static_cast< float* >(queue.enqueueMapBuffer(
                     weightsBuf, CL_TRUE, CL_MAP_WRITE, 0, m_weights.size() * sizeof(float)));
                    float* outputsHost = static_cast< float* >(queue.enqueueMapBuffer(
                     outputsBuf, CL_TRUE, CL_MAP_WRITE, 0, currentOutputs.size() * sizeof(float)));
                    float* deltasHost = static_cast< float* >(queue.enqueueMapBuffer(
                     deltasBuf, CL_TRUE, CL_MAP_WRITE, 0, m_deltas.size() * sizeof(float)));

                    std::copy(nextDeltas.begin(), nextDeltas.end(), nextDeltasHost);
                    std::copy(m_weights.begin(), m_weights.end(), weightsHost);
                    std::copy(currentOutputs.begin(), currentOutputs.end(), outputsHost);

                    queue.enqueueUnmapMemObject(nextDeltasBuf, nextDeltasHost);
                    queue.enqueueUnmapMemObject(weightsBuf, weightsHost);
                    queue.enqueueUnmapMemObject(outputsBuf, outputsHost);
                    queue.enqueueUnmapMemObject(deltasBuf, deltasHost);

                    cl::Kernel kernel{ocl.program, "calc_hidden_deltas"};
                    kernel.setArg(0, nextDeltasBuf);
                    kernel.setArg(1, weightsBuf);
                    kernel.setArg(2, outputsBuf);
                    kernel.setArg(3, deltasBuf);
                    float mom = momentum(Var{}, Var{});
                    kernel.setArg(4, sizeof(float), &mom);
                    cl_uint currentSize = static_cast< cl_uint >(size());
                    cl_uint nextSize = static_cast< cl_uint >(affectedLayer.size());
                    cl_uint inputsPerNeuron = static_cast< cl_uint >(inputs());
                    kernel.setArg(5, sizeof(cl_uint), &currentSize);
                    kernel.setArg(6, sizeof(cl_uint), &nextSize);
                    kernel.setArg(7, sizeof(cl_uint), &inputsPerNeuron);

                    queue.enqueueNDRangeKernel(kernel,
                                               cl::NullRange,
                                               cl::NDRange(size()),
                                               cl::NullRange);

                    float* resultDeltas = static_cast< float* >(queue.enqueueMapBuffer(
                     deltasBuf, CL_TRUE, CL_MAP_READ, 0, m_deltas.size() * sizeof(float)));
                    std::copy(resultDeltas,
                              resultDeltas + m_deltas.size(),
                              m_deltas.begin());
                    queue.enqueueUnmapMemObject(deltasBuf, resultDeltas);
                } catch(const cl::Error& e) {
                    std::cerr << "OpenCL error: " << e.what() << std::endl;
                    auto& self = *this;
                    auto& outputFunc = std::get< 0 >(m_activationFunctions);
                    self.for_each(
                     [&self, &affectedLayer, &momentum, &outputFunc](auto i, auto& currentNeuron) {
                         Var sum{};
                         affectedLayer.for_each([&sum, &i, &affectedLayer](auto j, auto& neuron) {
                             auto affectedDelta = affectedLayer.getDelta(j.value);
                             auto affectedWeight = neuron.getWeight(i.value);
                             sum += affectedDelta * affectedWeight;
                         });

                         self.setDelta(i.value,
                                       momentum(self.getDelta(i.value),
                                                sum * outputFunc.derivate(
                                                       currentNeuron.getOutput())));
                     });
                }
            }

            Var getAccumulatedGradient(std::size_t neuronId, std::size_t inputIdx) const {
                return m_accumulatedWeightGradients[neuronId][inputIdx];
            }

            Var getAccumulatedBiasGradient(std::size_t neuronId) const {
                return m_accumulatedBiasGradient[neuronId];
            }

            void resetGradients() {
                for(std::size_t i = 0; i < size(); ++i) {
                    std::fill(m_accumulatedWeightGradients[i].begin(),
                              m_accumulatedWeightGradients[i].end(),
                              Var{});
                    m_accumulatedBiasGradient[i] = Var{};
                }
                syncGradientsToGPU();
            }

            template< typename Prototype, typename MomentumFunc >
            void calculateDeltasAndAccumulateGradients(const Prototype& prototype,
                                                       MomentumFunc momentum) {
                if(!m_weightsBuffer()) {
                    calculateDeltas(prototype, momentum);
                    accumulateGradients();
                    return;
                }
                try {
                    using namespace cl;
                    auto& ocl = OpenCLProgram::instance();
                    const auto& defaultDevice = ocl.devices.front();

                    auto& self = *this;
                    std::vector< float > outputs(size());
                    for(std::size_t i = 0; i < size(); ++i) {
                        outputs[i] = self[i].getOutput();
                    }
                    auto& expected = std::get< 1 >(prototype);

                    cl::CommandQueue queue(ocl.context, defaultDevice);

                    float* outputsHost = static_cast< float* >(queue.enqueueMapBuffer(
                     m_outputsBuffer, CL_TRUE, CL_MAP_WRITE, 0, outputs.size() * sizeof(float)));
                    std::copy(outputs.begin(), outputs.end(), outputsHost);
                    queue.enqueueUnmapMemObject(m_outputsBuffer, outputsHost);

                    float* expectedHost = static_cast< float* >(queue.enqueueMapBuffer(
                     m_expectedBuffer, CL_TRUE, CL_MAP_WRITE, 0, expected.size() * sizeof(float)));
                    std::copy(expected.begin(), expected.end(), expectedHost);
                    queue.enqueueUnmapMemObject(m_expectedBuffer, expectedHost);

                    syncInputsToGPU();
                    syncDeltasToGPU();

                    cl::Kernel kernel{ocl.program,
                                      "calc_output_deltas_and_gradients"};
                    kernel.setArg(0, m_outputsBuffer);
                    kernel.setArg(1, m_expectedBuffer);
                    kernel.setArg(2, m_inputsBuffer);
                    kernel.setArg(3, m_deltasBuffer);
                    kernel.setArg(4, m_weightGradsBuffer);
                    kernel.setArg(5, m_biasGradsBuffer);
                    float mom = momentum(Var{}, Var{});
                    kernel.setArg(6, sizeof(float), &mom);
                    cl_uint sz = static_cast< cl_uint >(size());
                    kernel.setArg(7, sizeof(cl_uint), &sz);
                    cl_uint inpPerNeuron = static_cast< cl_uint >(inputs());
                    kernel.setArg(8, sizeof(cl_uint), &inpPerNeuron);

                    queue.enqueueNDRangeKernel(kernel,
                                               cl::NullRange,
                                               cl::NDRange(size()),
                                               cl::NullRange);

                    float* resultDeltas = static_cast< float* >(queue.enqueueMapBuffer(
                     m_deltasBuffer, CL_TRUE, CL_MAP_READ, 0, m_deltas.size() * sizeof(float)));

                    std::copy(resultDeltas,
                              resultDeltas + m_deltas.size(),
                              m_deltas.begin());

                    queue.enqueueUnmapMemObject(m_deltasBuffer, resultDeltas);
                } catch(const cl::Error& e) {
                    std::cerr << "OpenCL error: " << e.what() << std::endl;
                    calculateDeltas(prototype, momentum);
                    accumulateGradients();
                }
            }

            template< typename AffectedLayer, typename MomentumFunc >
            void calculateHiddenDeltasAndAccumulateGradients(AffectedLayer& affectedLayer,
                                                             MomentumFunc momentum) {
                if(!m_weightsBuffer()) {
                    calculateHiddenDeltas(affectedLayer, momentum);
                    accumulateGradients();
                    return;
                }
                try {
                    using namespace cl;
                    auto& ocl = OpenCLProgram::instance();
                    const auto& defaultDevice = ocl.devices.front();

                    auto& nextDeltas = affectedLayer.deltas();

                    cl::CommandQueue queue(ocl.context, defaultDevice);

                    Buffer nextDeltasBuf(ocl.context,
                                         CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                         nextDeltas.size() * sizeof(float));

                    float* nextDeltasHost = static_cast< float* >(queue.enqueueMapBuffer(
                     nextDeltasBuf, CL_TRUE, CL_MAP_WRITE, 0, nextDeltas.size() * sizeof(float)));
                    std::copy(nextDeltas.begin(), nextDeltas.end(), nextDeltasHost);
                    queue.enqueueUnmapMemObject(nextDeltasBuf, nextDeltasHost);

                    syncInputsToGPU();
                    syncDeltasToGPU();

                    cl::Kernel kernel{ocl.program,
                                      "calc_hidden_deltas_and_gradients"};
                    kernel.setArg(0, nextDeltasBuf);
                    kernel.setArg(1, m_weightsBuffer);
                    kernel.setArg(2, m_outputsBuffer);
                    kernel.setArg(3, m_inputsBuffer);
                    kernel.setArg(4, m_deltasBuffer);
                    kernel.setArg(5, m_weightGradsBuffer);
                    kernel.setArg(6, m_biasGradsBuffer);
                    float mom = momentum(Var{}, Var{});
                    kernel.setArg(7, sizeof(float), &mom);
                    cl_uint currentSize = static_cast< cl_uint >(size());
                    cl_uint nextSize = static_cast< cl_uint >(affectedLayer.size());
                    cl_uint inputsPerNeuron = static_cast< cl_uint >(inputs());
                    kernel.setArg(8, sizeof(cl_uint), &currentSize);
                    kernel.setArg(9, sizeof(cl_uint), &nextSize);
                    kernel.setArg(10, sizeof(cl_uint), &inputsPerNeuron);

                    queue.enqueueNDRangeKernel(kernel,
                                               cl::NullRange,
                                               cl::NDRange(size()),
                                               cl::NullRange);

                    float* resultDeltas = static_cast< float* >(queue.enqueueMapBuffer(
                     m_deltasBuffer, CL_TRUE, CL_MAP_READ, 0, m_deltas.size() * sizeof(float)));

                    std::copy(resultDeltas,
                              resultDeltas + m_deltas.size(),
                              m_deltas.begin());

                    queue.enqueueUnmapMemObject(m_deltasBuffer, resultDeltas);
                } catch(const cl::Error& e) {
                    std::cerr << "OpenCL error: " << e.what() << std::endl;
                    calculateHiddenDeltas(affectedLayer, momentum);
                    accumulateGradients();
                }
            }

            void accumulateGradients() {
                if(!m_weightsBuffer()) {
                    for_each([this](auto i, auto& neuron) {
                        const auto inputsNumber = neuron.size();
                        auto delta = m_deltas[i.value];
                        for(std::size_t j = 0; j < inputsNumber; j++) {
                            auto input = neuron[j].value;
                            m_accumulatedWeightGradients[i.value][j] += input * delta;
                        }
                        m_accumulatedBiasGradient[i.value] += delta;
                    });
                    return;
                }
                try {
                    using namespace cl;
                    auto& ocl = OpenCLProgram::instance();
                    const auto& defaultDevice = ocl.devices.front();

                    cl::CommandQueue queue(ocl.context, defaultDevice);

                    syncInputsToGPU();
                    syncDeltasToGPU();

                    cl::Kernel kernel{ocl.program, "accumulate_gradients"};
                    kernel.setArg(0, m_inputsBuffer);
                    kernel.setArg(1, m_deltasBuffer);
                    kernel.setArg(2, m_weightGradsBuffer);
                    kernel.setArg(3, m_biasGradsBuffer);
                    cl_uint sz = static_cast< cl_uint >(size());
                    kernel.setArg(4, sizeof(cl_uint), &sz);
                    cl_uint inpPerNeuron = static_cast< cl_uint >(inputs());
                    kernel.setArg(5, sizeof(cl_uint), &inpPerNeuron);

                    queue.enqueueNDRangeKernel(kernel,
                                               cl::NullRange,
                                               cl::NDRange(size()),
                                               cl::NullRange);
                } catch(const cl::Error& e) {
                    std::cerr << "OpenCL error: " << e.what() << std::endl;
                    for_each([this](auto i, auto& neuron) {
                        const auto inputsNumber = neuron.size();
                        auto delta = m_deltas[i.value];
                        for(std::size_t j = 0; j < inputsNumber; j++) {
                            auto input = neuron[j].value;
                            m_accumulatedWeightGradients[i.value][j] += input * delta;
                        }
                        m_accumulatedBiasGradient[i.value] += delta;
                    });
                }
            }

            const Var& getDelta(std::size_t neuronId) const {
                return m_deltas[neuronId];
            }

            void setDelta(std::size_t neuronId, const Var& delta) {
                m_deltas[neuronId] = delta;
            }

            std::vector< Var >& deltas() {
                return m_deltas;
            }
            const std::vector< Var >& deltas() const {
                return m_deltas;
            }

          private:
            using Base::bufferSize;
            using Base::m_dotProducts;
            using Base::m_inputs;
            using Base::m_weights;
            ActivationFunctions m_activationFunctions{};
            std::vector< Var > m_deltas;
            std::vector< Var > m_accumulatedBiasGradient;
            std::vector< std::vector< Var > > m_accumulatedWeightGradients;
            cl::Buffer m_weightsBuffer;
            cl::Buffer m_weightGradsBuffer;
            cl::Buffer m_biasGradsBuffer;
            cl::Buffer m_deltasBuffer;
            cl::Buffer m_outputsBuffer;
            cl::Buffer m_expectedBuffer;
        };

    } // namespace bp

} // namespace nn