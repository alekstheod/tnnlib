#pragma once

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
