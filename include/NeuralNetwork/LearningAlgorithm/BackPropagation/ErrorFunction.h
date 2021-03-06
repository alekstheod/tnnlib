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

#ifndef ERRORFUNCTION_H
#define ERRORFUNCTION_H
#include <cmath>

namespace nn {

    namespace bp {

        template< typename Var >
        struct SquaredError {
            template< typename Iterator, typename ProtoIterator >
            Var operator()(Iterator outputBegin, Iterator outputEnd, ProtoIterator protoBegin) {
                // Calculate error
                Var sum = 0;
                while(outputBegin != outputEnd) {
                    Var error = *outputBegin - *protoBegin;
                    sum += error * error;
                    outputBegin++;
                    protoBegin++;
                }

                return sum;
            }
        };


        template< typename Var >
        struct CrossEntropyError {
            template< typename Iterator, typename ProtoIterator >
            Var operator()(Iterator outputBegin, Iterator outputEnd, ProtoIterator protoBegin) {
                // Calculate error
                Var sum = 0;
                while(outputBegin != outputEnd) {
                    Var out = *outputBegin;
                    out = std::log(out);
                    sum -= out * (*protoBegin); // -
                                                // (1.f-*protoBegin)*utils::log(1.-*outputBegin);
                    outputBegin++;
                    protoBegin++;
                }

                return sum;
            }
        };
    } // namespace bp
} // namespace nn
#endif // ERRORFUNCTION_H
