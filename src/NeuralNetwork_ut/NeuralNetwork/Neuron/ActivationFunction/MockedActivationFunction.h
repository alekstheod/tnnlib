#ifndef EQUATIONMOCK_H
#define EQUATIONMOCK_H

#include <gmock/gmock.h>
#include <vector>

template<typename VarType>
class MockedActivationFunction{
public:
  typedef VarType Var;
  
public:
    MockedActivationFunction(){}
    MOCK_CONST_METHOD0_T( calcEquation, Var () );
    MOCK_CONST_METHOD1_T( calculateDerivate, Var ( const Var& input ) );
    MOCK_CONST_METHOD1_T( calcSum, Var ( const Var& start ) );
    
    template<typename It>
    Var calculateSum(It begin, It end, const Var& start)const{
      return calcSum(start);
    }
    
    template<typename It>
    Var calculateEquation(const Var& sum, It begin, It end)const{
      return calcEquation();
    }
    ~MockedActivationFunction(){}
};

#endif

