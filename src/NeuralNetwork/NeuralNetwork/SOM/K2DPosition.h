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

#ifndef K2DPOSITION_H
#define K2DPOSITION_H
#include <Utilities/Math/Math.h>
#include <boost/numeric/conversion/cast.hpp>

namespace nn
{

namespace kohonen
{

template< typename VarType, unsigned int rowSize >
class K2DPosition
{
public:
    typedef VarType Var;

private:
    Var m_x;
    Var m_y;

public:
    K2DPosition ( unsigned int id ) : m_x ( boost::numeric_cast<Var>(id % rowSize) ), m_y ( boost::numeric_cast<Var>(id / rowSize) ) {
    }
    
    K2DPosition( const K2DPosition& other ):m_x(other.m_x), m_y(other.m_y){}

    Var calculateDistance ( const K2DPosition& other )const {
        return sqrt ( utils::pow ( ( m_x - other.m_x ), 2 ) + utils::pow ( ( m_y - other.m_y ), 2 ) );
    }
    
    bool operator == (const K2DPosition& other )const{
      return (m_x == other.m_x && m_y == other.m_y);
    }
    
    unsigned int calculateId()const{
      return m_x+(m_y*rowSize);
    }

    ~K2DPosition() {}
};

}

}
#endif
