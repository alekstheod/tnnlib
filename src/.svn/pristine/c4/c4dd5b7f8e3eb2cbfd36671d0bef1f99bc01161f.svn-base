/**
*  Copyright (c) 2011, Alex Theodoridis
*  All rights reserved.

*  Redistribution and use in source and binary forms, with
*  or without modification, are permitted provided that the
*  following conditions are met:
*  Redistributions of source code must retain the above
*  copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above
*  copyright notice, this list of conditions and the following
*  disclaimer in the documentation and/or other materials
*  provided with the distribution.

*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
*  AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
*  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
*  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
*  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
*  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
*  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE,
*  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

#ifndef NeuronH
#define NeuronH

#include <map>
#include <NeuralNetwork/Neuron/ActivationFunction/IActivationFunction.h>
#include <NeuralNetwork/NNException.h>
#include <NeuralNetwork/Serialization/NeuronMemento.h>
#include <NeuralNetwork/Neuron/INeuron.h>
#include <Utilities/System/Time.h>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/bind.hpp>
#include <utility>

namespace nn {
/**
 * Neuron class.
 * Represent the neuron in the Neural layer.
 * Contains equation for output value calculation
 * and list of accepted inputs.
 */
namespace detail{
template<typename OutputFunctionType>
class Neuron {
public:
    typedef IActivationFunction< OutputFunctionType > OutputFunction;
    typedef typename OutputFunction::Var Var;
    typedef NeuronMemento<Var> Memento;

    typedef typename std::pair<Var, Var> Input;
    /// @brief a list of the inputs first is the weight, second is the value
    typedef typename std::vector< Input > Inputs;
    typedef typename Inputs::const_iterator iterator;

private:
    /**
     * @brief Instance of output calculation equation.
     * @brief The equation should be provided by implementation of IEquationFactory interface.
     */
    OutputFunction m_activationFunction;

    /**
     * @brief List of neurons inputs.
     */
    Inputs m_inputs;

    /**
     * @brief The neurons output.
     * @brief The output will be calculated with using the instance of calculation equation.
     */
    Var m_output;

    /**
     * @brief Neurons weight.
     * @brief Needed in order to improve the flexibility of neural network.
     */
    Var m_bias;

    /**
     * @brief Needed in order to calculate the neurons output value.
     */
    Var m_sum;

    /// @brief will create an vector with the initialized inputs.
    /// @param inputsNumber the number of inputs.
    /// @return a vector of initialized inputs.
    std::vector< std::pair< Var, Var> > initInputs(unsigned int inputsNumber)const {
        std::vector< std::pair< Var, Var> > result;
        result.reserve(inputsNumber);
        for( unsigned int i = 0; i < inputsNumber; i++ ) {
            result.push_back( std::pair<Var, Var>(utils::createRandom<Var>(1) , utils::createRandom<Var>(1)) );
        }

        return result;
    }

    /// @brief needed in order to calculate the neurons output.
    Var sumInput(const Input& input)const {
        return input.first * input.second;
    }

public:
    /**
     * Initialization constructor.
     * @param inputsNumber the number of inputs for current neuron.
     * @exception NNException thrown on object initialization failure.
     */
    Neuron ( unsigned int inputsNumber = 1 ) :  m_bias( utils::createRandom<Var>(1) ),
                                                m_output( boost::numeric_cast<Var>(0) ),
                                                m_sum( boost::numeric_cast<Var>(0) ),
                                                m_inputs(initInputs(inputsNumber)) 
    {
        if ( inputsNumber == 0 ) {
            throw NNException ( "Wrong argument inputsNumber==0", __FILE__, __LINE__ );
        }
    }

    /// @brief see @ref INeuron
    iterator begin()const {
        return m_inputs.begin();
    }

    /// @brief see @ref INeuron
    iterator end()const {
        return m_inputs.end();
    }

    /// @brief see @ref INeuron
    unsigned int size ( ) const {
        return m_inputs.size();
    }

    /// @brief see @ref INeuron
    Input& operator [] (unsigned int id) {
        return m_inputs[id];
    }

    /// @brief see @ref INeuron
    bool setWeight ( unsigned int weightId, const Var& weight ) {
        bool result = false;
        if( weightId < m_inputs.size() ) {
            m_inputs[weightId].first = weight;
            result = true;
        }

        return result;
    }

    /// @brief see @ref INeuron
    const Var& getBias()const {
        return m_bias;
    }

    /// @brief see @ref INeuron
    const Var& getWeight( unsigned int weightId )const {
        if ( weightId >= m_inputs.size() ) {
            throw NNException ( "Invalid weightId", __FILE__, __LINE__ );
        }

        return m_inputs[weightId].first;
    }

    /// @brief see @ref INeuron
    const Memento getMemento() const {
        Memento memento;
        memento.setInputs ( m_inputs );
        memento.setOutput ( m_output );
        memento.setBias ( m_bias );
        memento.setSum ( m_sum );
        return memento;
    }

    /// @brief see @ref INeuron
    void setMemento ( const Memento& memento ) {
        if ( memento.getInputs().size() != m_inputs.size() ) {
            throw NNException ( "Wrong argument memento, invalid number of inputs", __FILE__, __LINE__ );
        }

        m_inputs=memento.getInputs();
        m_output=memento.getOutput();
        m_bias=memento.getBias();
        m_sum=memento.getSum();
    }

    /// @brief see @ref INeuron
    void setInput ( unsigned int inputId, const Var& value ) {
        if ( inputId >= m_inputs.size() )
        {
            throw NNException ( "Wrong argument inputId", __FILE__, __LINE__ );
        }

        m_inputs[inputId].second = value;
    }

    /// @brief see @ref INeuron
    Var calcDotProduct()const {
        auto begin = boost::make_transform_iterator(m_inputs.cbegin(), boost::bind(&Neuron::sumInput, this, _1) );
        auto end = boost::make_transform_iterator(m_inputs.cend(), boost::bind(&Neuron::sumInput, this, _1) );
        return m_activationFunction.calculateSum(begin, end, m_bias);
    }

    /// @brief see @ref INeuron
    const Var& getOutput () const {
        return m_output;
    }

    /// @brief see @ref INeuron
    unsigned int getInputsNumber ()const {
        return m_inputs.size();
    }

    /// @brief see @ref INeuron
    void setBias ( Var weight ) {
        m_bias = weight;
    }

    /// @brief see @ref INeuron
    const Var& getNeuronWeight () const {
        return m_bias;
    }

    /// @brief see @ref INeuron
    template<typename Iterator>
    const Var& calculateOutput (Iterator begin, Iterator end) {
        m_output = m_activationFunction.calculateEquation ( calcDotProduct(), begin, end );
        return m_output;
    }

    template<typename Test>
    void supportTest(Test&);

    /**
     * Destructor.
     */
    ~Neuron () {}
};
}

template<template<class> class OutputFunctionType, typename VarType>
using Neuron = detail::Neuron< OutputFunctionType<VarType> >;

}

#endif
// kate: indent-mode cstyle; replace-tabs on; 
