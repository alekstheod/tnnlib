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
 * THIS SOFTWARE IS PROVIDED BY Alex Theodoridis <email> ''AS IS'' AND ANY
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

#ifndef INODE_H
#define INODE_H

namespace nn {

    namespace kohonen {

        template< typename NodeType >
        class INode {
          public:
            typedef NodeType Node;
            typedef typename NodeType::Var Var;
            typedef typename NodeType::Position Position;
            typedef typename NodeType::InputType InputType;

          private:
            NodeType m_node;

          public:
            template< unsigned int inputsNumber >
            struct rebindInputsNumber {
                typedef INode< typename NodeType::template rebindInputsNumber< inputsNumber >::Type > Type;
            };


          public:
            template< typename... Args >
            INode(Args... args) : m_node(args...) {
            }

            Var calculateDistance(const InputType& input) const {
                return m_node.calculateDistance(input);
            }

            void applyWeightModifications(Var learningRate, Var influence, const InputType& input) {
                m_node.applyWeightModifications(learningRate, influence, input);
            }

            const InputType& getWeights() const {
                return m_node.getWeights();
            }

            const Position& getPosition() const {
                return m_node.getPosition();
            }

            ~INode() {
            }
        };
    } // namespace kohonen
} // namespace nn


#endif // INODE_H
