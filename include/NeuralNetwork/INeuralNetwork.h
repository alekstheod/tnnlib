#pragma once

namespace nn {

    template< class NeuralNetwork >
    class INeuralNetwork {
      private:
        NeuralNetwork m_neuralNetwork;

      public:
        bool calculateOutputs(){};
        virtual ~INeuralNetwork() {
        }
    };
} // namespace nn
