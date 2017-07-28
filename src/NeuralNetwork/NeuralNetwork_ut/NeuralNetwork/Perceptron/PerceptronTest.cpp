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

#include "PerceptronTest.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Utilities/GMockUtils.h>

using ::testing::Return;
using ::testing::Ref;
using ::testing::_;
using ::testing::Mock;
using ::testing::InSequence;

using Perceptron = PerceptronTest::Perceptron;

PerceptronTest::PerceptronTest () {
}

PerceptronTest::~PerceptronTest () {
}

void PerceptronTest::SetUp () {
    m_inputs.resize (4, 0.f);
    m_neurons[0] = Neuron ();
    m_neurons[1] = Neuron ();
}

void PerceptronTest::TearDown () {
}

USING_SUPPORT_TEST_T_NN (PerceptronTest, TestCalculateOutputsForOneLayerPerceptron, nn, detail, Perceptron)
TEST_F (PerceptronTest, TestCalculateOutputsForOneLayerPerceptron) {
    Perceptron perceptron;
    perceptron.supportTest (*this);
    std::array< float, 2 > outputs = {0.f, 0.f};
    perceptron.calculate (m_inputs.begin (), m_inputs.end (), outputs.begin ());
    ASSERT_EQ (1.f, outputs[0]);
    ASSERT_EQ (2.f, outputs[1]);
}

SUPPORT_TEST_T (PerceptronTest, TestCalculateOutputsForOneLayerPerceptron, Perceptron) {
    InSequence dummy;
    EXPECT_CALL (*std::get< 0 > (m_layers), setInput (0, test.m_inputs[0])).Times (1);
    EXPECT_CALL (*std::get< 0 > (m_layers), setInput (1, test.m_inputs[1])).Times (1);
    EXPECT_CALL (*std::get< 0 > (m_layers), setInput (2, test.m_inputs[2])).Times (1);
    EXPECT_CALL (*std::get< 0 > (m_layers), setInput (3, test.m_inputs[3])).Times (1);
    EXPECT_CALL (*std::get< 0 > (m_layers), calculateOutputs (_)).Times (1);
    EXPECT_CALL (*std::get< 1 > (m_layers), calculateOutputs ()).Times (1);
    EXPECT_CALL (*std::get< 1 > (m_layers), end ()).WillOnce (Return (test.m_neurons.end ()));
    EXPECT_CALL (*std::get< 1 > (m_layers), begin ()).WillOnce (Return (test.m_neurons.begin ()));
    EXPECT_CALL (**test.m_neurons[0], getOutput ()).Times (1).WillRepeatedly (Return (1.f));
    EXPECT_CALL (**test.m_neurons[1], getOutput ()).Times (1).WillRepeatedly (Return (2.f));
}

/*
USING_SUPPORT_TEST_T_NN(PerceptronTest, TestCalculateOutputsForTwoLayersPerceptron, nn, detail, Perceptron)
TEST_F(PerceptronTest, TestCalculateOutputsForTwoLayersPerceptron)
{
    Perceptron perceptron;
    perceptron.supportTest(*this);
    std::array< float, 1 > outputs = { 0 };
    perceptron.calculate(m_inputs.begin(), m_inputs.end(), outputs.begin() );
    ASSERT_EQ( 1.f, outputs[0] );
}

SUPPORT_TEST_T(PerceptronTest, TestCalculateOutputsForTwoLayersPerceptron, Perceptron)
{
    InSequence dummy;

    EXPECT_CALL( *std::get<0>(m_layers),  setInput(0, test.m_inputs[0] ) ).Times(1);
    EXPECT_CALL( *std::get<0>(m_layers),  setInput(0, test.m_inputs[1] ) ).Times(1);
    EXPECT_CALL( *std::get<0>(m_layers),  setInput(0, test.m_inputs[2] ) ).Times(1);
    EXPECT_CALL( *std::get<0>(m_layers),  setInput(0, test.m_inputs[3] ) ).Times(1);
    EXPECT_CALL( *std::get<0>(m_layers),  calculateOutputs( _ ) ).Times(1);
    EXPECT_CALL( *std::get<1>(m_layers),  calculateOutputs() ).Times(1);
    EXPECT_CALL( *std::get<1>(m_layers),  end() ).Times(1).WillOnce(Return(neurons.begin()+1) );
    EXPECT_CALL( *std::get<1>(m_layers),  begin() ).Times(1).WillOnce(Return(neurons.begin()) );
    //EXPECT_CALL( **neurons[0],   getOutput() ).Times(1).WillOnce(Return( 1.f ) );
}*/