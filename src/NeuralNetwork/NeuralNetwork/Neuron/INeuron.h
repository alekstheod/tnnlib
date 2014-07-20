/*
    Copyright (c) 2013, Alex Theodoridis <email>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Alex Theodoridis <email> ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Alex Theodoridis <email> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef INEURON_H
#define INEURON_H

namespace nn
{

template<class Neuron>
class INeuron {
public:
    typedef typename Neuron::Var Var;
    typedef typename Neuron::OutputFunction OutputFunction;
    typedef typename Neuron::Memento Memento;
    typedef typename Neuron::Input Input;
    typedef Neuron NeuronType;

private:
    Neuron m_neuron;

public:
    INeuron() {}

    INeuron ( Neuron neuron ) :m_neuron ( neuron ) {}

    /**
     * @brief will calculate the output of the neuron.
     * @return the calculated value.
     */
    template<typename Iterator>
    const Var& calculateOutput (Iterator begin, Iterator end) {
        return m_neuron.calculateOutput(begin, end);
    }

    /**
    * @brief Will return the number of inputs for current neuron.
    * @return the number of inputs.
    */
    unsigned int size ( ) const {
        return m_neuron.size();
    }

    Input& operator [] (unsigned int id) {
        return m_neuron[id];
    }

    /**
     * @brief will return the output value of the current neuron.
     * @return a current output value of the neuron.
     */
    Var getOutput() const {
        return m_neuron.getOutput();
    }

    
    Var calcDotProduct()const {
        return m_neuron.calcDotProduct();
    }

    /**
     * @brief set the value to the given input
     * @param inputId the id of the input in which the value will be assigned.
     * @param value a value.
     */
    void setInput ( unsigned int inputId, const Var& value ) {
        m_neuron.setInput ( inputId, value );
    }
    
    /**
     *	Will set the inputs weight value.
     *	@param weightId the inputs identifier to set the new weight value.
     *	@param weigh a new weight value.
     */
    bool setWeight ( unsigned int weightId, const Var& weight ) {
        return m_neuron.setWeight ( weightId, weight );
    }

    const Var& getBias () const {
        return m_neuron.getBias();
    }

    void setBias ( Var weight ) {
       m_neuron.setBias ( weight );
    }

    const Var& getWeight (unsigned int weightId) const {
        return m_neuron.getWeight(weightId);
    }

    const Memento getMemento() const {
        return m_neuron.getMemento();
    }

    const Neuron* operator->() const {
        return &m_neuron;
    }

    Neuron* operator->() {
        return &m_neuron;
    }

    ~INeuron() {}
};

}

#endif // INEURON_H
