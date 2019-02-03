/*
 * Copyright (c) 2013, Alex Theodoridis <alekstheod@gmail.com>
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

#ifndef K2DNEIGHBOURHOOD_H
#define K2DNEIGHBOURHOOD_H

#include <SOM/IPosition.h>
#include <SOM/INode.h>
#include <vector>
#include <algorithm>
#include <boost/iterator/filter_iterator.hpp>
#include <math.h>

namespace nn {

    namespace kohonen {

        template< typename NodeType >
        class K2DNeighbourhood {
          public:
            typedef INode< NodeType > Node;
            typedef typename Node::Position Position;
            typedef typename Position::Var Var;

          public:
            template< unsigned int inputsNumber >
            struct rebindInputsNumber {
                typedef K2DNeighbourhood< typename Node::template rebindInputsNumber< inputsNumber >::Type::Node > Neighbourhood;
            };


          private:
            typedef typename std::vector< Node >::iterator viterator;
            typedef typename std::vector< Node >::const_iterator vconst_iterator;
            typedef IPosition< Position > IPos;

          private:
            IPos m_center;
            Var m_radius;
            std::pair< viterator, viterator > m_nodes;

          private:
            struct IsInNbhd {
                IPos m_center;
                Var m_radius;
                IsInNbhd(const IPos& center, const Var& radius)
                 : m_center(center), m_radius(radius) {
                }
                bool operator()(const Node& node) {
                    Var distance = node.getPosition().calculateDistance(m_center);
                    return (distance < m_radius);
                }
            };

          public:
            typedef boost::filter_iterator< IsInNbhd, viterator > iterator;
            typedef boost::filter_iterator< IsInNbhd, viterator > const_iterator;

          public:
            template< typename Iterator >
            K2DNeighbourhood(Position center, Var radius, Iterator begin, Iterator end)
             : m_center(center), m_radius(radius), m_nodes(begin, end) {
            }

            iterator begin() {
                return boost::make_filter_iterator< IsInNbhd >(IsInNbhd(m_center, m_radius),
                                                               m_nodes.first,
                                                               m_nodes.second);
            }

            iterator end() {
                return boost::make_filter_iterator< IsInNbhd >(IsInNbhd(m_center, m_radius),
                                                               m_nodes.second,
                                                               m_nodes.second);
            }

            ~K2DNeighbourhood() {
            }
        };
    } // namespace kohonen
} // namespace nn

#endif
