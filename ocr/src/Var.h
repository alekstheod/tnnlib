/*
 * Copyright (c) 2014, alekstheod <email>
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
 * THIS SOFTWARE IS PROVIDED BY alekstheod <email> ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL alekstheod <email> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef VAR_H
#define VAR_H

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>

#include <array>
#include <functional>

namespace ocr {

    template< unsigned int size >
    struct Var {
        typedef std::array< float, size > row;
        boost::array< row, size > m_rows;

        Var() : Var(0.f) {
        }

        Var(float val) {
            for(auto r = m_rows.begin(); r != m_rows.end(); r++) {
                std::fill(r->begin(), r->end(), val);
            }
        }

        Var(const Var& other) {
            for(unsigned int i = 0; i < size; i++) {
                std::copy(other[i].begin(), other[i].end(), m_rows[i].begin());
            }
        }

        Var& operator=(const Var& other) {
            if(&other != this) {
                for(unsigned int i = 0; i < size; i++) {
                    std::copy(other[i].begin(), other[i].end(), m_rows[i].begin());
                }
            }

            return *this;
        }

        Var operator*(const Var& var) const {
            return perform(var, std::multiplies< float >());
        }

        Var operator-(const Var& var) const {
            return perform(var, std::minus< float >());
        }

        Var operator/(const Var& var) const {
            return perform(var, std::divides< float >());
        }

        row& operator[](unsigned int i) {
            return m_rows[i];
        }

        const row& operator[](unsigned int i) const {
            return m_rows[i];
        }

        float calculateAvg() const {
            float sum = 0;
            for(unsigned int i = 0; i < size; i++) {
                for(unsigned int j = 0; j < size; j++) {
                    sum += m_rows[i][j];
                }
            }

            return (sum / (size * size));
        }

        float max() const {
            float maximum = m_rows[0][0];
            unsigned int k = 0, n = 0;
            for(unsigned int i = 0; i < size; i++) {
                for(unsigned int j = 0; j < size; j++) {
                    if(m_rows[i][j] > maximum) {
                        k = i;
                        n = j;
                        maximum = m_rows[i][j];
                    }
                }
            }

            return maximum;
        }

        bool operator>(float other) {
            bool result = false;
            float sum = 0;
            for(unsigned int i = 0; i < size && !result; i++) {
                for(unsigned int j = 0; j < size && !result; j++) {
                    if(m_rows[i][j] > other) {
                        result = true;
                    }
                }
            }

            return result;
        }

        ~Var() {
        }

      private:
        template< typename Operation >
        Var perform(const Var& var, Operation op) const {
            ocr::Var< size > result;
            for(unsigned int i = 0; i < size; i++) {
                for(unsigned int j = 0; j < size; j++) {
                    result[i][j] = op(m_rows[i][j], var[i][j]);
                }
            }

            return result;
        }

        friend class boost::serialization::access;
        template< class Archive >
        void serialize(Archive& ar, const unsigned int version) {
            ar& BOOST_SERIALIZATION_NVP(m_rows);
        }
    };

    template< unsigned int size >
    Var< size > operator+(const Var< size >& left, const Var< size >& right) {
        Var< size > result(0.f);
        for(unsigned int i = 0; i < size; i++) {
            std::transform(left[i].begin(),
                           left[i].end(),
                           right[i].begin(),
                           result.m_rows[i].begin(),
                           std::plus< float >());
        }

        return result;
    }

    template< unsigned int size >
    Var< size >& operator+=(Var< size >& left, const Var< size >& right) {
        for(unsigned int i = 0; i < size; i++) {
            std::transform(left[i].begin(),
                           left[i].end(),
                           right[i].begin(),
                           left[i].begin(),
                           std::plus< float >());
        }

        return left;
    }
} // namespace ocr

#endif // VAR_H
