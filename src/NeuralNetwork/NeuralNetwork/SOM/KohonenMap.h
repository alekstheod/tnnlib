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

#ifndef KOHONENMAP_H
#define KOHONENMAP_H

#include <vector>
#include <algorithm>
#include <type_traits>
#include <NeuralNetwork/SOM/INeighbourhood.h>
#include <NeuralNetwork/NNException.h>
#include <Utilities/System/Time.h>
#include <boost/numeric/conversion/cast.hpp>
#include <cmath>

namespace nn {

    namespace kohonen {

        template < typename NeighbourhoodType, unsigned int numberOfNodes, unsigned int inputsNumber > class KohonenMap {
            public:
            typedef INeighbourhood< typename NeighbourhoodType::template rebindInputsNumber< inputsNumber >::Neighbourhood > Neighbourhood;

            private:
            typedef typename Neighbourhood::Node NodeType;

            public:
            typedef typename NodeType::template rebindInputsNumber< inputsNumber >::Type Node;
            typedef typename Node::InputType InputType;
            typedef typename Node::Var Var;
            typedef typename Node::Position Position;
            typedef typename std::vector< Node >::const_iterator iterator;

            private:
            std::vector< Node > m_nodes;

            private:
            virtual void applyWeightModifications (Node& node, Position bestCandidate, Var radius, Var learningRate, const InputType& input) {
                Var distance = bestCandidate.calculateDistance (node.getPosition ());

                if (radius > 0.001f) {
                    Var influence (std::exp ((-(distance * distance) / (boost::numeric_cast< Var > (2) * radius * radius))));
                    node.applyWeightModifications (learningRate, influence, input);
                }
            }

            template < typename InputIterator > void executeEpoch (InputIterator begin, InputIterator end, Var radius, Var learningRate) {
                int randNumb = utils::createRandom< int > (std::distance (begin, end));
                auto currentInput (*(begin + randNumb));
                unsigned int nodeId = 0;
                Position bestCandidate (calculateBestCandidate (currentInput, nodeId));
                Neighbourhood nbhd (bestCandidate, radius, m_nodes.begin (), m_nodes.end ());

                std::for_each (nbhd.begin (), nbhd.end (), std::bind (&KohonenMap::applyWeightModifications, this, std::placeholders::_1, bestCandidate, radius, learningRate, currentInput));
            }

            /**
             * @brief will calculate the best candidate for the given input.
             * @param input the input
             * @param nodeId[out] the node id
             * @return the position of the best candidate.
             */
            Position calculateBestCandidate (const InputType& input, unsigned int& nodeId) const {
                Var distance (m_nodes[0].calculateDistance (input));
                Position currentPosition (m_nodes[0].getPosition ());
                nodeId = 0;
                for (auto pNode = m_nodes.cbegin (); pNode != m_nodes.cend (); ++pNode) {
                    Var newDistance = pNode->calculateDistance (input);
                    if (newDistance < distance) {
                        distance = newDistance;
                        nodeId = std::distance (m_nodes.cbegin (), pNode);
                        currentPosition = pNode->getPosition ();
                    }
                }

                return currentPosition;
            }

            public:
            KohonenMap () {
                static_assert (numberOfNodes > 0, "Invalid number of nodes");
                static_assert (inputsNumber >= 2, "Invalid number of inputs");
                for (unsigned int i = 0; i < numberOfNodes; i++) {
                    m_nodes.push_back (Node (Position (i)));
                }
            }

            /// @brief will classify the inputs to the different clusters by using the kohonen algorithm.
            /// @param iterationsNumber number of iterations which will be used in classification.
            /// @param learningRate the learning rate. This variable affects the learning mechanism.
            /// @param initRadius the initial radius of the neighbourhood.
            /// @param inputsData the inputs set used in the learning process.
            /// @return true if the inputsData is not empty, false otherwise.
            template < typename InputIterator >
            bool calculateWeights (InputIterator begin, InputIterator end, unsigned int iterationsNumber, const Var& learningRate, Var initRadius) {
                bool result = false;
                if (begin != end) {
                    result = true;
                    Var timeConstant = initRadius / std::log (initRadius);
                    Var currentLearningRate = learningRate;
                    for (unsigned int i = 0; i < iterationsNumber; i++) {
                        Var radius = Var (initRadius * Var (std::exp (-boost::numeric_cast< Var > (i) / timeConstant)));
                        executeEpoch (begin, end, radius, currentLearningRate);
                        currentLearningRate = learningRate * Var (std::exp (-boost::numeric_cast< Var > (i) / boost::numeric_cast< Var > (iterationsNumber)));
                    }
                }

                return result;
            }

            /// @brief stl compatible interface, to iterate over the nodes.
            /// @return will return the iterator which points to the first node in a map.
            iterator begin () const {
                return m_nodes.cbegin ();
            }

            /// @brief stl compatible interface, to iterate over the nodes.
            /// @return will return the iterator which points to the end of the list with the nodes.
            iterator end () const {
                return m_nodes.cend ();
            }

            ~KohonenMap () {
            }
        };
    }
}

#endif // KOHONENMAP_H
