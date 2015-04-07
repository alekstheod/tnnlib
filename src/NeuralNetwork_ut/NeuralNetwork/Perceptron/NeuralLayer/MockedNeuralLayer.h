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

#ifndef NEURALLAYERMOCK_H
#define NEURALLAYERMOCK_H
#include <gmock/gmock.h>
#include <NeuralNetwork/Serialization/NeuralLayerMemento.h>
#include <NeuralNetwork/Neuron/MockedNeuron.h>
#include <NeuralNetwork/Neuron/INeuron.h>
#include <NeuralNetwork/Neuron/ActivationFunction/MockedActivationFunction.h>
#include <NeuralNetwork/INeuralLayer.h>
#include <memory>
#include <boost/bind.hpp>

template<typename VarType, std::size_t neurons, std::size_t inputs = 1>
class MockedNeuralLayer
{
public:
    typedef typename nn::NeuralLayerMemento<VarType> Memento;
    typedef VarType Var;
    typedef typename std::vector<Var>::const_iterator VarIterator;
    typedef typename std::vector< std::pair<Var, Var> >::const_iterator NeuronInputIterator;
    typedef typename nn::INeuron< MockedNeuron< MockedActivationFunction<Var> > > Neuron;
    typedef typename std::array<Neuron, neurons>::const_iterator const_iterator;
    typedef typename std::array<Neuron, neurons>::iterator iterator;
    typedef typename std::array<Neuron, neurons>::reverse_iterator reverse_iterator;
    typedef typename std::array<Neuron, neurons>::const_reverse_iterator const_reverse_iterator;
    typedef typename nn::INeuralLayer<MockedNeuralLayer>& INeuralLayer;
    BOOST_STATIC_CONSTEXPR std::size_t CONST_NEURONS_NUMBER = neurons;
    BOOST_STATIC_CONSTEXPR std::size_t CONST_INPUTS_NUMBER = neurons;

    template<typename NewType>
    struct rebindNeuron {
        typedef MockedNeuralLayer type;
    };
    
    template<typename V>
    struct rebindVar{
      typedef MockedNeuralLayer< V, neurons, inputs > type;
    };
    
    template< std::size_t new_inputs>
    struct rebindInputs{
      typedef MockedNeuralLayer<VarType, neurons, new_inputs> type;
    };    

public:
    /// Need this because the googlemock does not allow to copy the mock objects.
    class Mock {
    public:
        MOCK_CONST_METHOD0_T(begin, const_iterator () );
        MOCK_CONST_METHOD0_T(end, const_iterator () );
        MOCK_METHOD0_T(begin, iterator () );
        MOCK_METHOD0_T(end, iterator () );
        MOCK_METHOD0_T(rbegin, reverse_iterator () );
        MOCK_METHOD0_T(rend, reverse_iterator () );
        MOCK_CONST_METHOD0_T(rbegin, const_reverse_iterator () );
        MOCK_CONST_METHOD0_T(rend, const_reverse_iterator () );
        MOCK_CONST_METHOD0_T(getInputsNumber, unsigned int() );
        MOCK_METHOD2_T( setInput, void ( unsigned int, const Var& ) );
        MOCK_CONST_METHOD1_T( getOutput, Var ( unsigned int ) );
        MOCK_METHOD1_T( getNeuron, const Neuron& (unsigned int ) );
        MOCK_CONST_METHOD2_T(getInputWeight, const Var& ( unsigned int, unsigned int ) );
        MOCK_CONST_METHOD0_T(getMemento, const Memento() );
        MOCK_METHOD1_T( setMemento, void ( const Memento& ) );
        MOCK_METHOD1_T( calculateOutputs, void ( int ) );
        MOCK_METHOD0_T( calculateOutputs, void () );
    };

public:
    unsigned int m_inputsNumber;

private:
    std::shared_ptr< Mock > m_mock;

public:
    MockedNeuralLayer():MockedNeuralLayer(0) {}
    MockedNeuralLayer(unsigned int inputsNumber) : m_inputsNumber( inputsNumber ),  m_mock(new Mock) {
    }

    unsigned int size()const {
        return 10;
    }

    Mock& operator* () {
        return (*m_mock.get());
    }

    const_iterator find ( unsigned int neuronId ) const {
    }

    const_iterator begin() const {
        return m_mock->begin();
    }

    const_iterator end() const {
        return m_mock->end();
    }

    iterator begin()  {
        return m_mock->begin();
    }

    iterator end()  {
        return m_mock->end();
    }

    reverse_iterator rbegin() {
        return m_mock->rbegin();
    }

    reverse_iterator rend() {
        return m_mock->rend();
    }

    const_reverse_iterator rbegin()const {
        return m_mock->rbegin();
    }

    const_reverse_iterator rend()const {
        return m_mock->rend();
    }

    const Neuron& operator [] ( unsigned int id ) const {
        return m_mock->getNeuron(id);
    }

    const Var& getInputWeight ( unsigned int neuronId, unsigned int weightId ) const {
        return m_mock->getInputWeight(neuronId, weightId);
    }

    const Memento getMemento() const {
        return m_mock->getMemento();
    }

    void setMemento ( const Memento& memento ) {
        return m_mock->setMemento(memento);
    }

    void setInput( unsigned int inputId, const Var& value ) {
        m_mock->setInput(inputId, value);
    }

    template<typename Layer>
    void calculateOutputs ( Layer& nextLayer ) {
        return m_mock->calculateOutputs(0);
    }

    void calculateOutputs () {
        return m_mock->calculateOutputs();
    }


    ~MockedNeuralLayer() {}
};

#endif // NEURALLAYERMOCK_H
