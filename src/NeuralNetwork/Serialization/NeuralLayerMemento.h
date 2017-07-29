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

#ifndef NEURALLAYERMEMENTO_H
#define NEURALLAYERMEMENTO_H

#include <NeuralNetwork/Serialization/NeuronMemento.h>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <vector>

namespace nn {

    template < typename NeuronMemento, std::size_t neuronsNumber > class NeuralLayerMemento {
        private:
        using Container = std::vector< NeuronMemento >;
        Container m_neurons;

        private:
        friend class boost::serialization::access;

        template < class Archive > void serialize (Archive& ar, const unsigned int version) {
            ar& BOOST_SERIALIZATION_NVP (m_neurons);
        }

        public:
        NeuralLayerMemento () : m_neurons (neuronsNumber) {
        }

        void setNeurons (const Container& neurons) {
            m_neurons = neurons;
        }

        const Container& getNeurons () const {
            return m_neurons;
        }

        unsigned int getNeuronsNumber () const {
            return m_neurons.size ();
        }
    };
}

#endif // NEURALLAYERMEMENTO_H
