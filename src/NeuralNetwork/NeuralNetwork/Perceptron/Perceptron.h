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

#ifndef PerceptronH
#define PerceptronH

#include <NeuralNetwork/Serialization/PerceptronMemento.h>
#include <NeuralNetwork/INeuralLayer.h>
#include <NeuralNetwork/NNException.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <Utilities/MPL/Tuple.h>
#include <vector>
#include <cassert>
#include <tuple>
#include <type_traits>

namespace nn {

/*! \class Perceptron
 *  \briefs Contains an input neurons layer one output and one or more hidden layers.
 */
namespace detail {

/// workaround for VS++ compilation.
template <typename Var, typename T>
struct rebindOne { typedef typename T::template rebindVar<Var>::type type; };

template<typename Var, typename T>
struct rebindVar;

template<typename Var, typename... T>
struct rebindVar<Var, std::tuple<T...> > {
  typedef typename std::tuple< typename rebindOne<Var, T>::type... > type;
};

template<std::size_t FirstInputs, typename RebindedTuple, typename Tuple>
struct RebindInputsHelper;

template<std::size_t FirstInputs, typename RebindedTuple, typename FirstLayer, typename... Layers>
struct RebindInputsHelper<FirstInputs, RebindedTuple, std::tuple<FirstLayer, Layers...>>
{                           
    typedef typename utils::push_back<typename FirstLayer::template rebindInputs<FirstInputs>::type, RebindedTuple>::type CurrentRebindedTuple;            
    typedef typename std::conditional<sizeof...(Layers) == 0,
	    CurrentRebindedTuple,
	    typename RebindInputsHelper<FirstLayer::CONST_NEURONS_NUMBER, CurrentRebindedTuple, std::tuple<Layers...>>::type>::type type;
};

template<std::size_t FirstInputs, typename RebindedTuple, typename... Layers>
struct RebindInputsHelper<FirstInputs, RebindedTuple, std::tuple<Layers...>>
{
    typedef RebindedTuple type;
};

template<std::size_t FirstInputs, typename Tuple>
struct rebindInputs
{
    static_assert(std::tuple_size<Tuple>::value >= 1, "Invalid");
    typedef typename RebindInputsHelper<FirstInputs, std::tuple<>, Tuple>::type type;
};



template<typename VarType, typename LayerTypes>
class Perceptron {
private:
    typedef typename rebindVar<VarType, LayerTypes>::type TmplLayers;

public:
    BOOST_STATIC_CONSTEXPR std::size_t CONST_LAYERS_NUMBER = std::tuple_size<TmplLayers>::value;
    using InputLayerType = typename std::tuple_element<0, TmplLayers>::type;

    BOOST_STATIC_CONSTEXPR std::size_t CONST_INPUTS_NUMBER = InputLayerType::CONST_NEURONS_NUMBER;
    using Layers = typename rebindInputs< CONST_INPUTS_NUMBER, TmplLayers >::type;
    using OutputLayerType = typename std::tuple_element<CONST_LAYERS_NUMBER - 1, Layers>::type;

    BOOST_STATIC_CONSTEXPR std::size_t CONST_OUTPUTS_NUMBER = OutputLayerType::CONST_NEURONS_NUMBER;

    typedef VarType Var;
    template<template <class> class Layer>
    struct wrap {
        typedef typename utils::rebind_tuple<Layer, Layers>::type NewLayers;
        typedef detail::Perceptron<VarType, NewLayers > type;
    };

    /// @brief Memento type.
    typedef PerceptronMemento<Var> Memento;

private:
    /*!
     * Hidden layers.
     */
    Layers  m_layers;

    struct SetMemento {
        int& m_position;
        std::vector< NeuralLayerMemento<Var> > m_layers;
        SetMemento(int& position, std::vector< NeuralLayerMemento<Var> > layers):m_position(position), m_layers(layers) {}
        template<typename T>
        void operator()(T& layer) {
            layer.setMemento(m_layers[m_position]);
            m_position++;
        }
    };

    struct GetMemento {
        std::vector<NeuralLayerMemento<Var> >& m_layers;
        GetMemento( std::vector<NeuralLayerMemento<Var> >& layers ) : m_layers(layers) {}
        template<typename T>
        void operator()(const T& layer) {
            m_layers.push_back(layer.getMemento());
        }
    };

    template<unsigned int index>
    void calculate(Layers&, int) {}

    template<unsigned int index>
    void calculate(Layers& layers, bool) {
        std::get<index>(layers).calculateOutputs( std::get<index+1>(layers) );

        enum { enable = (index < CONST_LAYERS_NUMBER - 2) };
        typedef typename std::conditional< enable, bool, int >::type ArgType;
        calculate<index+1>( layers,  ArgType(0));
    }

public:
    /*!
     *
     */
    Perceptron()
    {
        static_assert( std::tuple_size< Layers >::value > 1 , "Invalid number of layers, at least two layers need to be set" );
    }
    
    Perceptron( const Layers& layers ):m_layers(layers){}
    
    Layers& layers() {
        return m_layers;
    }

    void setMemento( const Memento& memento )
    {
        auto layers = memento.getLayers();
        int position = 0;
        utils::for_each(m_layers, SetMemento(position, layers) );
    }

    Memento getMemento()const
    {
        typename std::vector< NeuralLayerMemento<Var> > layers;
        layers.reserve( CONST_LAYERS_NUMBER );
        utils::for_each_c(m_layers, GetMemento(layers) );
        PerceptronMemento<Var> memento;
        memento.setLayers(layers);
        return memento;
    }

    /*!
     * @brief this method will calculate the outputs of perceptron.
     * @param begin is the iterator which is pointing to the first input
     * @param end the iterator which is pointing to the last input
     * @param out the output iterator where the results of the calculation will be stored.
     */
    template<typename Iterator, typename OutputIterator>
    void calculate(Iterator begin, Iterator end, OutputIterator out)
    {
        unsigned int inputId = 0;
        while( begin != end ) {
            std::get<0>(m_layers).setInput(inputId, *begin);
            begin++;
            inputId++;
        }

        calculate<0>(m_layers, true);
        typedef typename std::tuple_element< CONST_LAYERS_NUMBER -1 , Layers>::type OutputLayer;
        std::get< CONST_LAYERS_NUMBER -1 >(m_layers).calculateOutputs();
        std::transform( std::get< CONST_LAYERS_NUMBER -1 >(m_layers).begin(),
                        std::get< CONST_LAYERS_NUMBER -1 >(m_layers).end(),
                        out,
                        std::bind(&OutputLayer::Neuron::getOutput, std::placeholders::_1)
                      );
    }

    /**
     * @brief only for the testing purpose.
     * @brief please don't use this function.
     */
    template<typename Test>
    void supportTest(Test&);

    /*!
     * Destructor
     */
    ~Perceptron() {
    }
};

}

template<typename VarType, typename... NeuralLayers>
using Perceptron = detail::Perceptron<VarType, std::tuple<NeuralLayers...> >;

}

#endif
