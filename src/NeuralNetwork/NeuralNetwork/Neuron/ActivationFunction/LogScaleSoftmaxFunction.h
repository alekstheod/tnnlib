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

#ifndef LogScaleSoftmaxFunction_H
#define LogScaleSoftmaxFunction_H

#include <Utilities/Math/Math.h>
#include <functional>
#include <numeric>
#include <utility>
#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/concept_check.hpp>

namespace nn
{

template<typename VarType>
class LogScaleSoftmaxFunction
{
public:
    typedef VarType Var;
    template<typename V>
    struct rebindVar {
        typedef LogScaleSoftmaxFunction<V> type;
    };
    
public:
    LogScaleSoftmaxFunction() {}
    ~LogScaleSoftmaxFunction() {}

    template<typename Iterator>
    Var calculate ( const Var& sum, Iterator begin, Iterator end )const {
        Var sum2 = boost::numeric_cast<Var>(0.f);
        while(begin != end ) {
            sum2 += *begin ;
            begin++;
        }

        return sum/sum2;
    }

    template<typename Iterator>
    Var sum (Iterator begin, Iterator end, const Var& start)const {
	Var max = *begin; //*max_element(begin, end); // doesn't work with transform_iterator...
	Iterator i = begin;
	while(i != end ){
	  if( *i > max ){
	    max = *i;
	  }
	  
	  i++;
	}
	
	Var sum = boost::numeric_cast<Var>(0.f);
	while(begin != end ){
	  sum += utils::exp(*begin - max);
	  begin++;
	}
	
	sum = max + utils::log(sum);
        return sum;
    }

    Var delta ( const Var& output, const Var& expectedOutput)const {
        return ( output - expectedOutput );
    }
};

}

#endif // LogScaleSoftmaxFunction_H
