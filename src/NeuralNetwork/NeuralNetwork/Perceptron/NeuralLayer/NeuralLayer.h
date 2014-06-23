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

#ifndef NeuralLayerH
#define NeuralLayerH

#include <NeuralNetwork/Neuron/INeuron.h>
#include <NeuralNetwork/Serialization/NeuralLayerMemento.h>
#include <NeuralNetwork/INeuralLayer.h>
#include <NeuralNetwork/NNException.h>
#include <boost/iterator/transform_iterator.hpp>
#include <functional>
#include <boost/bind.hpp>
#include <boost/predef.h>
#include <boost/bind/placeholders.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <algorithm>
#include <vector>
#include <array>

namespace nn
{

namespace detail {
/**
 * Represent the NeuralLayer in perceptron.
 */
template<class NeuronType,unsigned int neuronsNumber>
class NeuralLayer
{
public:
    typedef INeuron<NeuronType> Neuron;
    typedef typename Neuron::Var Var;
    typedef typename Neuron::Memento NeuronMemento;
    typedef NeuralLayerMemento<Var> Memento;
    typedef typename std::array< Neuron, neuronsNumber >::const_iterator const_iterator;
    typedef typename std::array< Neuron, neuronsNumber>::iterator iterator;
    typedef typename std::array< Neuron, neuronsNumber>::reverse_iterator reverse_iterator;
    typedef typename std::array< Neuron, neuronsNumber>::const_reverse_iterator const_reverse_iterator;

    template<template <class> class NewType>
    struct rebind {
        typedef NeuralLayer< NewType<NeuronType>, neuronsNumber > type;
    };

    template<typename NewType>
    struct rebindNeuron {
        typedef NeuralLayer< NewType, neuronsNumber > type;
    };

    template<typename VarType>
    struct rebindVar{
      typedef NeuralLayer< typename NeuronType::template rebindVar<VarType>::type , neuronsNumber > type;
    };
    
    BOOST_STATIC_CONSTEXPR unsigned int CONST_NEURONS_NUMBER = neuronsNumber;

private:
    /**
     * A list of the neurons.
     */
    typename std::array< Neuron, neuronsNumber > m_neurons;

    /**
     * Number of available inputs for the current layer.
     */
    unsigned int m_inputsNumber;
public:
    /**
     * Constructor will initialize the layer by the given inputs number and neurons number.
     * @param inputsNumber the number of inputs of the layer.
     * @param neuronsNumber the number of the neurons in the layer.
     * @throw NNException in case of invalid arguments.
     */
    NeuralLayer (unsigned int inputsNumber = 1) : m_inputsNumber(inputsNumber) {
        static_assert(neuronsNumber > 0, "Invalid template argument neuronsNumber == 0");
        std::generate ( m_neurons.begin(), m_neurons.end(), [inputsNumber]() {
            return Neuron ( inputsNumber );
        } );
    }

    /**
    * @see {INeuralLayer}
    */
    const_iterator cbegin() const {
        return m_neurons.cbegin();
    }

    /**
    * @see {INeuralLayer}
    */
    const_iterator cend() const {
        return m_neurons.cend();
    }

    /**
    * @see {INeuralLayer}
    */
    iterator begin() {
        return m_neurons.begin();
    }

    /**
    * @see {INeuralLayer}
    */
    iterator end() {
        return m_neurons.end();
    }

    /**
     * @see {INeuralLayer}
     */
    unsigned int size() const {
        return m_neurons.size();
    }

    /**
    * @see {INeuralLayer}
    */
    const Neuron& operator [] ( unsigned int id ) const {
        return m_neurons[id];
    }

    /**
     * @see {INeuralLayer}
     */
    unsigned int getInputsNumber() const {
        return m_inputsNumber;
    }

    reverse_iterator rbegin() {
        return m_neurons.rbegin();
    }

    reverse_iterator rend() {
        return m_neurons.rend();
    }

    const_reverse_iterator rbegin()const {
        return m_neurons.rbegin();
    }

    const_reverse_iterator rend()const {
        return m_neurons.rend();
    }


    /**
     * @see {INeuralLayer}
     */
    void setInput ( unsigned int inputId, const Var& value ) {
        std::for_each ( m_neurons.begin(), m_neurons.end(), std::bind ( &Neuron::setInput, std::placeholders::_1, inputId, value ) );
    }

    const Var& getBias( unsigned int neuronId )const {
        if ( neuronId >= m_neurons.size() ) {
            throw NNException ( "Invalid argument neuronId", __FILE__, __LINE__ );
        }

        return m_neurons[neuronId].getBias();
    }

    /**
    * @see {INeuralLayer}
    */
    const Var& getInputWeight ( unsigned int neuronId, unsigned int weightId ) const {
        if ( neuronId >= m_neurons.size() ) {
            throw NNException ( "Invalid argument neuronId", __FILE__, __LINE__ );
        }

        return m_neurons[neuronId].getWeight ( weightId );
    }

    /**
     * @see {INeuralLayer}
     */
    const Memento getMemento() const {
        Memento memento;
        std::vector< NeuronMemento > neurons;
        std::transform ( m_neurons.begin(), m_neurons.end(), std::back_inserter ( neurons ), std::bind ( &Neuron::getMemento, std::placeholders::_1 ) );
        memento.setNeurons ( neurons );
        return memento;
    }

    /**
     * @see {INeuralLayer}
     */
    void setMemento ( const NeuralLayerMemento<Var>& memento ) {
        if ( memento.getNeuronsNumber() < 1 ) {
            throw nn::NNException("Invalid argument memento", __FILE__, __LINE__ );
        }

        auto neurons=memento.getNeurons();
        std::vector< Neuron > internalNeurons;
        std::transform ( neurons.begin(), neurons.end(), std::back_inserter ( internalNeurons ), [] ( NeuronMemento& m ) {
            Neuron neuron ( m.getInputsNumber() );
            neuron->setMemento ( m );
            return neuron;
        } );

	/// TODO no exception guarantee, please fix as soon as possible.
        //std::swap(m_neurons, internalNeurons);
	std::copy(internalNeurons.begin(), internalNeurons.end(), m_neurons.begin() );
    }

    /**
     * @see {INeuralLayer}
     */
    Var getOutput ( unsigned int outputId ) const {
        if ( outputId >= m_neurons.size() ) {
            throw NNException ( "Wrong outputId", __FILE__, __LINE__ );
        }

        return m_neurons[outputId].getOutput();
    }

    /**
     * @see {INeuralLayer}
     */
    template<typename Layer>
    void calculateOutputs ( Layer& nextLayer ) {
        for ( unsigned int i = 0; i < m_neurons.size(); i++ ) {
            auto begin = boost::make_transform_iterator(m_neurons.begin(), std::bind(&Neuron::calcDotProduct, std::placeholders::_1));
            auto end = boost::make_transform_iterator(m_neurons.end(), std::bind(&Neuron::calcDotProduct, std::placeholders::_1) );
            nextLayer.setInput ( i, m_neurons[i].calculateOutput( begin ,end ) );
        }
    }

    /**
     * @see {INeuralLayer}
     */
    void calculateOutputs() {
        auto begin = boost::make_transform_iterator(m_neurons.begin(), boost::bind(&Neuron::calcDotProduct, ::_1));
        auto end = boost::make_transform_iterator(m_neurons.end(), boost::bind(&Neuron::calcDotProduct, ::_1));
        using IteratorType = decltype ( begin );
        std::for_each ( m_neurons.begin(), m_neurons.end(), std::bind ( &Neuron::template calculateOutput<IteratorType>, std::placeholders::_1, begin, end ) );
    }

    /**
     *
     */
    ~NeuralLayer() {
    }
};

}

template<
	 template<template<class> class, class> class NeuronType,
         template<class> class ActivationFunctionType,
	 unsigned int size, 
	 typename Var = float
         >
using NeuralLayer = detail::NeuralLayer<NeuronType<ActivationFunctionType, Var >, size >;

}

#endif
