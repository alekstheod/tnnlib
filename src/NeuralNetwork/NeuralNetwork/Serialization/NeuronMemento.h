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

#ifndef NEURONMEMENTO_H
#define NEURONMEMENTO_H
#include <map>
#include <array>
#include <utility>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/utility.hpp>
#include <NeuralNetwork/Neuron/Input.h>

namespace nn {

/**
* @author alekstheod
* Represents the Neuron's memento (state)
* class. The instance of this class is enough in order
* to restore the Neuron's state.
*/
template<class Var, std::size_t inputsNumber>
class NeuronMemento {
private:
    /**
     * The list of the neuron's inputs.
     */
    using Container = std::array< nn::Input<Var>, inputsNumber >;
    Container m_inputs;

    /**
     * Value of the neuron's weight.
     */
    Var m_bias;

private:
    typedef nn::Input<Var> Input;
    friend class boost::serialization::access;
        
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP(m_bias);
        ar & BOOST_SERIALIZATION_NVP(m_inputs);
    }
    
public:
    /**
     * Will set the given inputs list.
     * The given inputs list should contain at least 1 element
     * in order to be assigned to the inputs list member variable.
     * @return true if succeed, false otherwise.
     */
    bool setInputs ( const Container& inputs ) {
        bool result = false;
        m_inputs = inputs;
        return result;
    }

    /**
     * Will set the given value to the
     * neuron weight member variable.
     * @param weight the value to be set.
     */
    void setBias ( const Var& weight ) {
        m_bias = weight;
    }

    /**
    * Will return the list of assigned inputs.
    * @return the list of assigned inputs.
    */
    const Container& getInputs() const {
        return m_inputs;
    }

    /**
     * Will return the neuron's weight value.
     * @return the neuron's weight value.
     */
    const Var& getBias() const {
        return m_bias;
    }
};

}

#endif // NEURONMEMENTO_H
// kate: indent-mode cstyle; indent-width 4; replace-tabs on; 
