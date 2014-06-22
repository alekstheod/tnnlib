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
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <vector>
#include <utility>

namespace nn {

/**
* @author alekstheod
* Represents the Neuron's memento (state)
* class. The instance of this class is enough in order
* to restore the Neuron's state.
*/
template<class Var>
class NeuronMemento {
private:
    /**
     * The list of the neuron's inputs.
     */
    std::vector< std::pair<Var, Var> > m_inputs;

    /**
     * Value of the neuron's weight.
     */
    Var m_bias;

    /**
     * The output value of the neuron.
     */
    Var m_output;

    /**
    * the value of the calculated inputs sum.
    */
    Var m_sum;

private:
    friend class boost::serialization::access;
        
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP(m_output);
        ar & BOOST_SERIALIZATION_NVP(m_sum);
        ar & BOOST_SERIALIZATION_NVP(m_bias);
        ar & BOOST_SERIALIZATION_NVP(m_inputs);
    }
    
public:
    /**
     * Empty constructor.
     * Will initialize the object.
     * Will set the default number of
     * inputs to 1.
     */
    NeuronMemento() : m_inputs(1, std::pair<Var, Var>( Var(0.0f), Var(0.0f) ) ) {
    }

    /**
     * Will set the given inputs list.
     * The given inputs list should contain at least 1 element
     * in order to be assigned to the inputs list member variable.
     * @return true if succeed, false otherwise.
     */
    bool setInputs ( const std::vector< std::pair<Var, Var> >& inputs ) {
        bool result = false;
        if ( !inputs.empty() ) {
            m_inputs = inputs;
        }

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

    unsigned int getInputsNumber() const {
        return m_inputs.size();
    }

    /**
    * Will return the list of assigned inputs.
    * @return the list of assigned inputs.
    */
    const std::vector< std::pair<Var, Var> >& getInputs() const {
        return m_inputs;
    }

    /**
     * Will return the neuron's weight value.
     * @return the neuron's weight value.
     */
    const Var& getBias() const {
        return m_bias;
    }

    /**
    * Will return the last calculated sum.
    * @return sum.
    */
    const Var& getSum() const {
        return m_sum;
    }

    /**
     * Will set the value to the member sum variable.
     * @param sum the value.
     */
    void setSum ( const Var& sum ) {
        m_sum = sum;
    }

    /**
    * Will set a value to the output member variable.
    * @param output the value.
    */
    void setOutput ( const Var& output ) {
        m_output = output;
    }

    const Var& getOutput() const {
        return m_output;
    }

    /**
     * Destructor.
     */
    ~NeuronMemento() {
    }

};

}

#endif // NEURONMEMENTO_H
// kate: indent-mode cstyle; indent-width 4; replace-tabs on; 
