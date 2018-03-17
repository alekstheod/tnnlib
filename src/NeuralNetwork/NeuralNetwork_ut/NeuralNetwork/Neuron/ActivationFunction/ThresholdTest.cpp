/*
 * Copyright (c) 2015, <copyright holder> <email>
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
 * THIS SOFTWARE IS PROVIDED BY <copyright holder> <email> ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <copyright holder> <email> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "ThresholdTest.h"
#include <NeuralNetwork/Neuron/ActivationFunction/MockedActivationFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/Threshold.h>
#include <array>

typedef MockedActivationFunction< float > Func;
using ::testing::Return;


INSTANTIATE_TEST_CASE_P(TestThreshold,
                        ThresholdTest,
                        ::testing::Values(Param(0.1f, 0.f),
                                          Param(0.f, 0.f),
                                          Param(0.01f, 0.f),
                                          Param(0.18f, 0.f),
                                          Param(0.19f, 0.f),
                                          Param(-0.1f, 0.f),
                                          Param(0.20f, 0.f),
                                          Param(0.21f, 1.f),
                                          Param(0.99f, 1.f),
                                          Param(0.6f, 1.f),
                                          Param(0.8f, 1.f),
                                          Param(1.1f, 1.f),
                                          Param(0.5f, 1.f)));


TEST_P(ThresholdTest, TestThreshold) {
    nn::Threshold< Func, 20 > func;
    std::array< float, 3 > inputs;
    auto param = GetParam();
    EXPECT_CALL(*func, calcEquation()).WillOnce(Return(param.activation));
    ASSERT_EQ(param.result, func.calculate(0, inputs.begin(), inputs.end()));
}
