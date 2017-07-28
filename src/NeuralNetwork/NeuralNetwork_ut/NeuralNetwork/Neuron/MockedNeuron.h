/*
 * Copyright (c) 2014, alekstheod <email>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY alekstheod <email> ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL alekstheod <email> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef NeuronMock_H
#define NeuronMock_H

#include <gmock/gmock.h>
#include <NeuralNetwork/Neuron/ActivationFunction/IActivationFunction.h>
#include <NeuralNetwork/Serialization/NeuronMemento.h>
#include <utility>
#include <memory>

template < typename OutputFunctionType, std::size_t inputsNumber > class MockedNeuron {
    public:
    typedef typename nn::IActivationFunction< OutputFunctionType > OutputFunction;
    typedef typename OutputFunction::Var Var;
    typedef typename nn::NeuronMemento< Var, inputsNumber > Memento;
    typedef typename std::pair< Var, Var > Input;

    template < typename OutputFunc > struct rebind { typedef MockedNeuron< OutputFunc, inputsNumber > type; };

    public:
    class Mock;

    private:
    std::shared_ptr< Mock > m_mock;

    public:
    class Mock {
        public:
        MOCK_METHOD2_T (setInput, void(unsigned int, const Var&));
        MOCK_CONST_METHOD0_T (getOutput, Var ());
        MOCK_CONST_METHOD0_T (calculateOutput, const Var&());
    };

    unsigned int size () const {
        return inputsNumber;
    }

    using TMock = MockedNeuron< OutputFunctionType, inputsNumber >;
    TMock& operator= (TMock other) {
        m_mock = other.m_mock;
        return *this;
    }

    void setInput (unsigned int id, const Var& value) {
        m_mock->setInput (id, value);
    }

    Var getOutput () const {
        return m_mock->getOutput ();
    }

    template < typename Iterator > const Var& calculateOutput (Iterator begin, Iterator end) {
        return m_mock->calculateOutput ();
    }

    Mock& operator* () {
        return (*m_mock.get ());
    }

    void clear () {
        m_mock.reset ();
    }

    MockedNeuron () : m_mock (new Mock) {
    }
};


#endif // NeuronMock_H
