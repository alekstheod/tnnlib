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

#ifndef TanhFunction_H
#define TanhFunction_H

#include <Utilities/Math/Math.h>
#include <functional>
#include <numeric>
#include <utility>
#include <NeuralNetwork/Neuron/ActivationFunction/IActivationFunction.h>
#include <boost/numeric/conversion/cast.hpp>

namespace nn
{

template<class VarType>
class TanhFunction
{
public:
    typedef VarType Var;
    template<typename V>
    struct rebindVar {
        typedef TanhFunction<V> type;
    };
    
public:
    TanhFunction() {}
    ~TanhFunction() {}

    //2 / (1 + exp(-2 * x)) - 1
    template<typename Iterator>
    Var calculateEquation ( const Var& sum, Iterator begin, Iterator end )const {
        return boost::numeric_cast<Var>(2.f) / ( boost::numeric_cast<Var>(1.f) + utils::exp ( boost::numeric_cast<Var>(-2.f)*sum ) ) - boost::numeric_cast<Var>(1.f);
    }
    
    template<typename Iterator>
    Var calculateSum(Iterator begin, Iterator end, const Var& start)const {
        return std::accumulate( begin, end, start );
    }

    /**
     * Will calculate the delta of given equation for the input value.
     * @param output the output value of the neuron.
     * @return calculated differential for given input value.
     */
    Var calculateDelta ( const Var& output, const Var& expectedOutput ) const {
        return ( output - expectedOutput ) * calculateDerivate(output);
    }

    Var calculateDerivate ( const Var& output)const {
        return boost::numeric_cast<Var>(1.f) - output*output;
    }
};

}

#endif // SOFTMAXFUNCTION_H
