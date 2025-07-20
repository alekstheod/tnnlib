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

typedef utils::StateMachine<
                utils::Transition< State1, finished, State2 >,
                utils::Transition< State2, finished, State1 >,
                utils::Transition< State2, repeat, State2 >,
                utils::Transition< State1, repeat, State1 >
                > StateMachine;
*/

#ifndef STATEMACHINE_H
#define STATEMACHINE_H
#include <memory>

namespace utils {

    template< typename CurState, typename EventType, typename NextState >
    struct Transition;

    namespace detail {

        template< typename... Transitions >
        class StateHolder;

        template< typename... Transitions >
        class StateMachine;

        template< typename CurState, typename Event, typename Unknown, typename... Transitions >
        struct NextState {
            typedef typename NextState< CurState, Event, Transitions... >::type type;
        };

        template< typename CurState, typename Event, typename Unknown, typename... Transitions >
        struct NextState< CurState, Event, Transition< CurState, Event, Unknown >, Transitions... > {
            typedef Unknown type;
        };


        /// @brief the state interface.
        template< typename... Transitions >
        class IState {
          private:
            StateHolder< Transitions... >& m_owner;

          private:
            virtual void ExecuteStepImpl(StateHolder< Transitions... >& stateMachine) = 0;

          protected:
            IState(StateHolder< Transitions... >& stateHolder)
             : m_owner(stateHolder) {};
            IState(IState&) = delete;

          public:
            void ExecuteStep(StateHolder< Transitions... >& stateMachine) {
                return ExecuteStepImpl(stateMachine);
            }

            virtual ~IState() {
            }
        };


        /// @brief the state holder class which keeps the current state of the
        /// state machine.
        template< typename... Transitions >
        class StateHolder {
          private:
            /// @brief the type of the state.
            typedef IState< Transitions... > State;

          private:
            template< typename >
            friend class Executor;
            std::unique_ptr< State > m_currentState;

          public:
            template< typename StateCreator >
            StateHolder(StateCreator creator) {
                m_currentState.reset(creator(*this));
            }

            ~StateHolder() {
            }

            /// @brief will set a new state to the state machine.
            /// @param newState instance of the new state.
            /// @return bool true is the given state is not NULL, false
            /// otherwise.
            template< class EventType, class StateType, typename... Args >
            void SendEvent(StateType* state, Args... args) {
                typedef
                 typename detail::NextState< StateType, EventType, Transitions... >::type NextState;
                m_currentState.reset(new NextState(*this, std::forward< Args >(args)...));
            }
        };


        template< typename StateHolder >
        class Executor {
          private:
            StateHolder m_stateHolder;

          protected:
            ~Executor() {
            }

          public:
            template< typename StateCreator >
            Executor(StateCreator creator) : m_stateHolder(creator) {
            }

            void ExecuteStep() {
                return m_stateHolder.m_currentState->ExecuteStep(m_stateHolder);
            }
        };
    } // namespace detail


    template< typename CurState, typename EventType, typename NextState >
    struct Transition {};

    template< typename... Transitions >
    class StateMachine
     : public detail::Executor< detail::StateHolder< Transitions... > > {
      private:
        typedef detail::Executor< detail::StateHolder< Transitions... > > Executor;

      public:
        typedef detail::IState< Transitions... > State;
        typedef detail::StateHolder< Transitions... > StateHolder;

      public:
        /// @brief Constructor will initialize the object.
        /// @param creator the creator function for the first state.
        template< typename StateCreator >
        StateMachine(StateCreator creator) : Executor(creator) {
        }
        ~StateMachine() {
        }
    };
} // namespace utils

#endif // STATEMACHINE_H
