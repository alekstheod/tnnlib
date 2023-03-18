#pragma once
#include "NeuralNetwork/Neuron/Input.h"

#include <cstdlib>

namespace nn {

    template< typename Var >
    class INeuron {
      public:
        using Input = nn::Input< Var >;

        /**
         * @brief Will return the number of inputs for current neuron.
         * @return the number of inputs.
         */
        virtual Input& operator[](std::size_t id) = 0;
        virtual const Input& operator[](std::size_t id) const = 0;

        /**
         * @brief will return the output value of the current neuron.
         * @return a current output value of the neuron.
         */
        virtual const Var& getOutput() const = 0;
        virtual Var calcDotProduct() const = 0;
        virtual const Var& getBias() const = 0;
        virtual void setBias(Var weight) = 0;
        virtual ~INeuron() = default;
    };
} // namespace nn
