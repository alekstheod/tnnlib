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

#include <NeuralNetwork/Perceptron/Perceptron.h>
#include <NeuralNetwork/LearningAlgorithm//BackPropagation/BepAlgorithm.h>
#include <Utilities/StrUtil/StrUtil.h>
#include <Utilities/Design/Singleton.h>
#include <NeuralNetwork/SOM/K2DPosition.h>
#include <NeuralNetwork/SOM/KohonenMap.h>
#include <NeuralNetwork/SOM/K2DNeighbourhood.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/TanhFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SoftmaxFunction.h>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <NeuralNetwork/SOM/KNode.h>
#include <Utilities/System/Time.h>
#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/Perceptron/NeuralLayer/NeuralLayer.h>
#include <Utilities/StrUtil/StrUtil.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <array>
#include <Utilities/Design/Factory.h>
#include <Utilities/Design/StateMachine.h>

using namespace nn;
using namespace bp;
using namespace std;
using namespace utils;

/*
 *
 */
struct finished {};
struct repeat {};
struct working {};

template<typename... Args>
struct Test {
    using TestType = typename utils::detail::NextState<int, finished, Args...>::type;
    TestType value;
};

int main ( int argc, char** argv )
{
    class State1;
    class State2;
    typedef utils::StateMachine<
	    utils::Transition<State1, finished, State2>,
	    utils::Transition<State2, repeat, State2>,
	    utils::Transition<State2, finished, State1> 
          > Machine;

    class State1 : public Machine::State {
    public:
        State1(Machine::StateHolder& stateHolder):Machine::State(stateHolder){}

    private:
        void ExecuteStepImpl( Machine::StateHolder& stateHolder) {
	    stateHolder.SendEvent<finished>(this);
        }
    };
    
    class State2 : public Machine::State {
    public:
        State2(Machine::StateHolder& stateHolder):Machine::State(stateHolder){}

    private:
        void ExecuteStepImpl( Machine::StateHolder& stateHolder) {
	    stateHolder.SendEvent<finished>(this);
        }
    };
    
    Machine m([](Machine::StateHolder& holder){return new State1(holder);});
    while(true){
      m.ExecuteStep();
    }

    return 0;
}

