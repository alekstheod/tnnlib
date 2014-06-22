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

 /** Example
  class State2;
  class State1;

  struct finished{};
  struct repeat{};
  struct working{};

  typedef utils::Entry<State1, finished, State2> Transition1;
  typedef utils::Entry<State2, finished, State1> Transition2;
  typedef utils::Entry<State1, repeat, State1> Transition3;
  typedef utils::Entry<State2, working, State2> Transition4;
  typedef TYPELIST_4( Transition1, Transition2, Transition3, Transition4 ) TransitionList;
  typedef utils::StateMachine<utils::Transitions< TransitionList >, bool, true > StateMachine;
 */

#ifndef STATEMACHINE_H
#define STATEMACHINE_H
#include <memory>
#include <Utilities/MPL/TypeList.h>

namespace utils
{

namespace priv
{

template<typename RetType,typename Transitions>
class StateHolder;

template<typename RetType,typename Transitions>
class StateMachine;


/// @brief the state interface.
template<typename RetType, typename Transitions>
class IState
{
private:
    StateHolder<RetType,Transitions>& m_owner;

private:
    virtual RetType ExecuteStepImpl ( StateHolder<RetType,Transitions>& stateMachine ) = 0;

protected:
    IState ( StateHolder<RetType, Transitions>& stateHolder ) : m_owner ( stateHolder ) {};

public:
    RetType ExecuteStep ( StateHolder<RetType,Transitions>& stateMachine ) {
        return ExecuteStepImpl ( stateMachine );
    }

    virtual ~IState() {}
};


/// @brief the state holder class which keeps the current state of the state machine.
template<typename RetType,class Transitions >
class StateHolder
{
private:
    /// @brief the type of the state.
    typedef IState<RetType, Transitions> State;
    typedef RetType ReturnType;

private:
    template<class, bool> friend class Executor;
    std::unique_ptr< State > m_currentState;

public:
    template< typename StateCreator >
    StateHolder( StateCreator creator){
      m_currentState.reset( creator( *this ) );
    }

    ~StateHolder() {}

    /// @brief will set a new state to the state machine.
    /// @param newState instance of the new state.
    /// @return bool true is the given state is not NULL, false otherwise.
    template< class EventType, class StateType>
    void SendEvent (StateType* state) {
        typedef typename Transitions::template NextState<StateType, EventType>::Result NextState;
        m_currentState.reset ( new NextState(*this) );
    }
    
    /// @brief will set a new state to the state machine.
    /// @param newState instance of the new state.
    /// @return bool true is the given state is not NULL, false otherwise.
    template< class EventType, class StateType, typename Arg1>
    void SendEvent (StateType* state, Arg1 arg1 ) {
        typedef typename Transitions::template NextState<StateType, EventType>::Result NextState;
        m_currentState.reset ( new NextState ( *this, arg1 ) );
    }
    
    /// @brief will set a new state to the state machine.
    /// @param newState instance of the new state.
    /// @return bool true is the given state is not NULL, false otherwise.
    template< typename EventType, class StateType, typename Arg1, typename Arg2>
    void SendEvent (StateType* state, Arg1 arg1, Arg2 arg2) {
        typedef typename Transitions::template NextState<StateType, EventType>::Result NextState;
        m_currentState.reset ( new NextState(*this, arg1, arg2) );
    }
};


template<typename StateHolder, bool includeExecutor>
class Executor {};

template<typename StateHolder>
class Executor<StateHolder, false >
{
protected:
    StateHolder m_stateHolder;

protected:
    ~Executor() {}

public:
    template< typename StateCreator>
    Executor( StateCreator creator) : m_stateHolder( creator) {
    }
};


template<typename StateHolder>
class Executor<StateHolder, true> : public Executor<StateHolder, false>
{
protected:
    typedef typename StateHolder::ReturnType RetType;
    typedef Executor<StateHolder, false> NullExecutor;

protected:
    ~Executor() {}

public:
    template< typename StateCreator >
    Executor( StateCreator creator) : NullExecutor( creator) {
    }

    RetType ExecuteStep() {
        return Executor<StateHolder, false>::m_stateHolder.m_currentState->ExecuteStep ( NullExecutor::m_stateHolder );
    }
};

}

/// @brief Generic state machine implementation.
template<class Transitions, typename RetType = void, bool includeExecutor = false>
class StateMachine : public priv::Executor< priv::StateHolder<RetType,Transitions>, includeExecutor >
{
private:
    typedef priv::Executor< priv::StateHolder<RetType,Transitions>, includeExecutor > Executor;

public:
    typedef priv::IState<RetType, Transitions> State;
    typedef priv::StateHolder<RetType, Transitions> StateHolder;
    typedef RetType ReturnType;

public:
    /// @brief Constructor will initialize the object.
    /// @param creator the creator function for the first state.
    template< typename StateCreator>
    StateMachine(StateCreator creator) : Executor(creator) {}
    ~StateMachine() { }
};

template<typename CurState, typename EventType, typename NextStateType>
struct Entry
{
  typedef CurState CurrentState;
  typedef NextStateType Result;
  typedef EventType Event;
};

template<class TList, class CurState, class EventType>
struct FindNextStep;

template<class TList, class CurState, class EventType>
struct FindNextStep
{
  typedef typename FindNextStep<typename TList::Tail, CurState, EventType>::Result Result;
};

template<class TList>
struct FindNextStep<TList, typename TList::Head::CurrentState, typename TList::Head::Event>
{
  typedef typename TList::Head::Result Result;
};

template<class CurState, class EventType>
struct FindNextStep<utils::NullType, CurState, EventType>
{
  typedef NullType Result;
};

template<class ListType>
struct Transitions 
{ 
  template< typename CurState, typename EventType>
  struct NextState
  {
    typedef typename FindNextStep<ListType, CurState, EventType>::Result Result;
  };
};

}

#endif // STATEMACHINE_H


