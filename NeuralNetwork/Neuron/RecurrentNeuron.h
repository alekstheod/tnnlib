#pragma once

#include "NeuralNetwork/Neuron/Neuron.h"

#include <cstddef>

namespace nn {

    template< template< class > typename OutputFunctionType, typename VarType = float, std::size_t inputsNumber = 1 >
    struct RecurrentNeuron
     : private nn::Neuron< OutputFunctionType, VarType, inputsNumber + 1 > {
        using Neuron = nn::Neuron< OutputFunctionType, VarType, inputsNumber + 1 >;
        using Neuron::calcDotProduct;
        using Neuron::cbegin;
        using Neuron::cend;
        using Neuron::getMemento;
        using Neuron::getOutput;
        using Neuron::setBias;
        using Neuron::setInput;
        using Neuron::setMemento;
        using Neuron::size;
        using Neuron::use;
        using Neuron::operator[];
        using typename Neuron::Input;
        using typename Neuron::Inputs;
        using typename Neuron::Memento;
        using typename Neuron::Var;

        template< std::size_t newSize >
        using resize = RecurrentNeuron< OutputFunctionType, Var, newSize + 1 >;

        template< typename Iterator >
        const Var& calculateOutput(const Var& dotProduct, Iterator begin, Iterator end) {
            auto output = Neuron::calculateOutput(dotProduct, begin, end);
            setInput(size() - 1, output);
            return getOutput();
        }
    };

} // namespace nn
