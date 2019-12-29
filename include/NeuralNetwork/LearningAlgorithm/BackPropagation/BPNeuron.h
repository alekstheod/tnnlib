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

#pragma once

#include <NeuralNetwork/Neuron/INeuron.h>

#include <MPL/TypeTraits.h>

#include <boost/numeric/conversion/cast.hpp>

#include <functional>
#include <type_traits>

namespace nn {

    namespace bp {

        template< typename Internal >
        class BPNeuron;

        // TODO please throw this out
        // This code fixes the blowing object when using
        // rebind functionality
        namespace detail {
            template< typename Internal >
            struct unwrapNeuron {
                using type = Internal;
            };

            template< typename Internal >
            struct unwrapNeuron< BPNeuron< Internal > > {
                using type = typename unwrapNeuron< Internal >::type;
            };
        } // namespace detail

        /*
         * Represent the back error propagation Neuron trainer.
         * This class holds a pointer to neuron which should
         * be trained with back error propagation algorithm.
         */
        template< class NeuronType >
        class BPNeuron
         : public INeuron< typename detail::unwrapNeuron< NeuronType >::type > {
          public:
            using Internal = typename detail::unwrapNeuron< NeuronType >::type;
            using Neuron = nn::INeuron< Internal >;
            using Var = typename Neuron::Var;
            using Memento = typename Neuron::Memento;
            using OutputFunction = typename Neuron::OutputFunction;
            using Input = typename Neuron::Input;

            template< typename EquationType >
            using use = BPNeuron< typename Internal::template use< EquationType > >;

            template< unsigned int inputs >
            using adjust = BPNeuron< typename Internal::template adjust< inputs > >;

          public:
            /**
             * @brief Default constructor will initialize the BPNeuron with 0.
             */
            BPNeuron() : m_delta(boost::numeric_cast< Var >(0.f)) {
            }

            /**
             * @brief Initialization constructor
             * @param neuron the pointer to the neuron which need to be trained
             */
            BPNeuron(unsigned int inputsNumber)
             : m_delta(boost::numeric_cast< Var >(0.f)), Neuron(inputsNumber) {
            }

            /**
             * @brief Will return the errors delta for the trained neuron.
             * @returns the error deltas value.
             */
            const Var& getDelta(void) const {
                return m_delta;
            }

            void setDelta(const Var& delta) {
                m_delta = delta;
            }

            void setMemento(const Memento& memento) {
                (*this)->setMemento(memento);
            }

            /*!
             *  Will calculate the differential value.
             *  @return the calculated value.
             */
            template< typename MomentumFunc >
            const Var& calculateDelta(const Var& expectedOutput, MomentumFunc momentum) {
                m_delta =
                 momentum(m_delta, m_outputFunction.delta(Neuron::getOutput(), expectedOutput));
                return m_delta;
            }

            const Var calculateDerivate() const {
                return m_outputFunction.derivate(Neuron::getOutput());
            }

          private:
            /**
             * Neurons error delta
             */
            Var m_delta;

            /**
             * Equation needed in order to calculate the differential value;
             */
            OutputFunction m_outputFunction;
        };
    } // namespace bp
} // namespace nn
