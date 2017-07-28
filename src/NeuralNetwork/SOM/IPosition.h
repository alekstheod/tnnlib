/*
 * Copyright (c) 2013, Alex Theodoridis <email>
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
 * THIS SOFTWARE IS PROVIDED BY Alex Theodoridis <alekstheod@gmail.com> ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Alex Theodoridis <email> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


#ifndef IPOSITION_H
#define IPOSITION_H

namespace nn {

    namespace kohonen {

        template < typename PositionType > class IPosition {
            public:
            typedef typename PositionType::Var Var;

            private:
            PositionType m_position;

            public:
            template < typename... Args > IPosition (Args... args) : m_position (args...) {
            }

            IPosition (const PositionType& pos) : m_position (pos) {
            }

            IPosition (const IPosition& other) : m_position (other.m_position) {
            }

            bool operator== (const IPosition& other) const {
                return m_position == other.m_position;
            }

            bool operator== (const PositionType& other) const {
                return m_position == other;
            }

            Var calculateDistance (const IPosition< PositionType >& other) const {
                return m_position.calculateDistance (other.m_position);
            }

            operator PositionType () {
                return m_position;
            }

            unsigned int calculateId () const {
                return m_position.calculateId ();
            }

            ~IPosition () {
            }
        };
    }
}
#endif
