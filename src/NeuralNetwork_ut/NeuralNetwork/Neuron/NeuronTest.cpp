#include <gtest/gtest.h>
#define TESTING
#include "NeuronTest.h"

#if defined(_MSC_VER)
// Some missed implementation
# include "gmock/gmock-spec-builders.h"
#endif

#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/Neuron/ActivationFunction/MockedActivationFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <boost/numeric/conversion/cast.hpp>
#include <array>
#include <Utilities/GMockUtils.h>

using ::testing::Return;
using ::testing::_;

NeuronTest::NeuronTest() {
}

NeuronTest::~NeuronTest() {
}

void NeuronTest::SetUp() {
  m_result = 0.f;
}

void NeuronTest::TearDown() {
}

typedef std::array<float, 4>::iterator Iterator;
typedef nn::SigmoidFunction<float> Equation;
typedef std::pair<float, float> Input;
typedef std::vector< Input > Inputs;
typedef Inputs::const_iterator InputIterator;
typedef nn::detail::Neuron<MockedActivationFunction<float>, 5, 1 > Neuron;
 

TEST_F(NeuronTest, TestSetInputWeight) {
    const unsigned int inputsNumber =  5;
    Neuron neuron;
    
    ASSERT_EQ( inputsNumber, neuron.getInputsNumber());

    for( unsigned int i =0; i< inputsNumber; i++) {
	// It's not important to use boost::numeric_cast
        float inputWeight = boost::numeric_cast<float>(rand());
        neuron.setWeight(i, inputWeight);
        ASSERT_EQ( inputWeight,  (neuron.begin()+i)->weight);
    }
}


USING_SUPPORT_TEST_T_NN(NeuronTest, TestCalculateOutput, nn, detail, Neuron)
TEST_F(NeuronTest, TestCalculateOutput) {
    const unsigned int inputsNumber =  rand()%1000 + 1;
    Neuron neuron;
    m_result = 3.f;
    neuron.supportTest(*this);
    std::array<float, 4> neurons;
    ASSERT_EQ(m_result,  neuron.calculateOutput( neurons.begin(), neurons.end() ) );
    ASSERT_EQ( m_result, neuron.getOutput());
}

SUPPORT_TEST_T(NeuronTest, TestCalculateOutput, Neuron){
  const int result = rand()%1000;
  EXPECT_CALL( **m_activationFunction,  calcSum( _) ).Times(1).WillRepeatedly(Return(10.f));
  EXPECT_CALL( **m_activationFunction,  calcEquation() ).Times(1).WillRepeatedly(Return(test.m_result));
}


TEST_F(NeuronTest, setInputTest) {
    const unsigned int inputsNumber =  rand()%1000 + 1;
    Neuron neuron;
    const float result = boost::numeric_cast<float>(rand()%1000);
    neuron.setInput(0, result);
    ASSERT_EQ(result, neuron.begin()->value);
}

TEST_F(NeuronTest, sizeTest) {
    Neuron neuron;
    ASSERT_EQ( 5, neuron.size() );
}

// kate: indent-mode cstyle; replace-tabs on; 
