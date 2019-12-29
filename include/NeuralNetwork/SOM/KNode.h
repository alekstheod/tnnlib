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
 * THIS SOFTWARE IS PROVIDED BY Alex Theodoridis <alekstheod@gmail.com> ''AS
 * IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Alex Theodoridis <email> BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef NODE_H
#define NODE_H

#include <NeuralNetwork/SOM/IPosition.h>

#include <System/Time.h>

#include <vector>
#include <array>
#include <iostream>
#include <cmath>

namespace nn {

    namespace kohonen {

        template< typename PositionType, unsigned int inputsNumber = 10 >
        class KNode {
          public:
            typedef IPosition< PositionType > Position;
            typedef typename Position::Var Var;
            typedef std::array< Var, inputsNumber > InputType;

          public:
            /// @brief rebind the current node to a different number of inputs.
            template< unsigned int numOfInputs >
            struct rebindInputsNumber {
                typedef KNode< PositionType, numOfInputs > Type;
            };

          private:
            InputType m_weights;
            Position m_position;

          public:
            /**
             * @brief constructor will initialize the object.
             * @param inputsNumber the number of the inputs for the current
             * object.
             * @param position a position inside the map.
             */
            KNode(const Position& position)
             : m_weights({Var(0.f), Var(0.f), Var(0.f)}), m_position(position) {
                Var weight(0.0f);
                for(unsigned int i = 0; i < inputsNumber; i++) {
                    weight = (Var(utils::createRandom< Var >(1)));
                    m_weights[i] = weight;
                }
            }

            /**
             * @brief Will calculate the distance of the input from the input
             * weight.
             * @param inputs the list of the input values.
             * @return the calulated euclidean distance.
             */
            Var calculateDistance(const InputType& input) const {
                Var result(0);
                for(unsigned int i = 0; i < m_weights.size(); i++) {
                    result += std::pow(m_weights[i] - input[i], 2);
                }

                return std::sqrt(result);
            }

            /// @brief getter for the weights
            /// @return the reference to the weights array.
            const InputType& getWeights() const {
                return m_weights;
            }

            /**
             * @brief will return the position of the Node in the map.
             * @return the position of the node.
             */
            const Position& getPosition() const {
                return m_position;
            }

            void applyWeightModifications(Var learningRate, Var influence, const InputType& input) {
                for(unsigned int i = 0; i < m_weights.size(); i++) {
                    m_weights[i] +=
                     learningRate * influence * (input[i] - m_weights[i]);
                }
            }
        };
    } // namespace kohonen
} // namespace nn

#endif // NODE_H
