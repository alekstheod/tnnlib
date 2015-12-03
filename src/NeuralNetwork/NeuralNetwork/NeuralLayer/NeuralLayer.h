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
#include <boost/iterator/transform_iterator.hpp>
#include <functional>
#include <boost/bind.hpp>
#include <boost/bind/placeholders.hpp>
#include <boost/config/suffix.hpp>
#include <algorithm>
#include <array>

namespace nn
{

namespace detail {
/**
 * Represent the NeuralLayer in perceptron.
 */
template<class NeuronType,
	 std::size_t neuronsNumber, 
	 std::size_t inputsNumber>
class NeuralLayer
{
public:
    typedef INeuron<typename NeuronType::template rebindInputs<inputsNumber>::type> Neuron;
    typedef typename Neuron::Var Var;
    typedef typename Neuron::Memento NeuronMemento;
    typedef NeuralLayerMemento<NeuronMemento, neuronsNumber> Memento;
    typedef typename std::vector<Neuron> Container;
				      
    typedef typename Container::const_iterator const_iterator;
    typedef typename Container::iterator iterator;
    typedef typename Container::reverse_iterator reverse_iterator;
    typedef typename Container::const_reverse_iterator const_reverse_iterator;

    template<template <class> class NewType>
    struct rebind {
        typedef NeuralLayer< NewType<NeuronType>, neuronsNumber, inputsNumber > type;
    };

    template<typename NewType, unsigned int inputs>
    struct rebindNeuron {
        typedef NeuralLayer< NewType, neuronsNumber, inputs > type;
    };
    
    template< unsigned int inputs>
    struct rebindInputs{
      typedef NeuralLayer<NeuronType, neuronsNumber, inputs> type;
    };

    template<typename VarType>
    struct rebindVar{
      typedef NeuralLayer< typename NeuronType::template rebindVar<VarType>::type , neuronsNumber, inputsNumber > type;
    };
    
    BOOST_STATIC_CONSTEXPR unsigned int CONST_NEURONS_NUMBER = neuronsNumber;
    BOOST_STATIC_CONSTEXPR unsigned int CONST_INPUTS_NUMBER = inputsNumber;
    
private:    
    /**
     * A list of the neurons.
     */
    Container m_neurons;   
    
public:
    NeuralLayer():m_neurons(neuronsNumber){}
  
    /**
     * Constructor will initialize the layer by the given inputs number and neurons number.
     */
    static_assert(neuronsNumber > 0, "Invalid template argument neuronsNumber == 0");
    static_assert(inputsNumber > 0, "Invalid template argument inputsNumber <= 1");

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
    Neuron& operator [] ( unsigned int id ) {
        return m_neurons[id];
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
        return m_neurons[neuronId].getBias();
    }

    /**
    * @see {INeuralLayer}
    */
    const Var& getInputWeight ( unsigned int neuronId, 
				unsigned int weightId ) const {
        return m_neurons[neuronId].getWeight ( weightId );
    }

    /**
     * @see {INeuralLayer}
     */
    const Memento getMemento() const {
        Memento memento;
        std::vector< NeuronMemento > neurons(CONST_NEURONS_NUMBER);
        std::transform ( m_neurons.begin(), 
			 m_neurons.end(), 
			 neurons.begin(), 
			 std::bind ( &Neuron::getMemento, std::placeholders::_1 ) );
	
        memento.setNeurons ( neurons );
        return memento;
    }

    /**
     * @see {INeuralLayer}
     */
    void setMemento ( const Memento& memento ) {
        auto neurons=memento.getNeurons();
        std::vector< Neuron > internalNeurons(CONST_NEURONS_NUMBER);
        std::transform ( neurons.begin(), neurons.end(), internalNeurons.begin(), [] ( NeuronMemento& m ) {
            Neuron neuron;
            neuron->setMemento ( m );
            return neuron;
        } );
	
	std::copy(internalNeurons.begin(), internalNeurons.end(), m_neurons.begin() );
    }

    /**
     * @see {INeuralLayer}
     */
    Var getOutput ( unsigned int outputId ) const {
        return m_neurons[outputId].getOutput();
    }

    /**
     * @see {INeuralLayer}
     */
    template<typename Layer>
    void calculateOutputs ( Layer& nextLayer ) {
	auto begin = boost::make_transform_iterator(m_neurons.begin(), boost::bind(&Neuron::calcDotProduct, _1));
	auto end = boost::make_transform_iterator(m_neurons.end(), boost::bind(&Neuron::calcDotProduct, _1));
        for ( unsigned int i = 0; i < m_neurons.size(); i++ ) {
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
};

}

template<
	 template<template<class> class, class, std::size_t, int> class NeuronType,
         template<class> class ActivationFunctionType,
	 std::size_t size,
	 std::size_t inputsNumber = 2,
	 int scaleFactor = 1,
	 typename Var = float
         >
using NeuralLayer = detail::NeuralLayer< NeuronType<ActivationFunctionType, Var, inputsNumber, scaleFactor >, size, inputsNumber >;

}

#endif
