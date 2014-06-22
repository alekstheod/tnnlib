#ifndef EQUATIONMOCK_H
#define EQUATIONMOCK_H

#include <gmock/gmock.h>
#include <vector>

template<typename VarType, typename Iterator>
class MockedActivationFunction{
public:
  typedef VarType Var;
  
public:
    MockedActivationFunction(){}
    MOCK_CONST_METHOD3_T( calculateEquation, Var ( const Var& sum, Iterator, Iterator) );
    MOCK_CONST_METHOD1_T( calculateDerivate, Var ( const Var& input ) );
    MOCK_CONST_METHOD1_T( calcSum, Var ( const Var& start ) );
    
    template<typename It>
    Var calculateSum(It begin, It end, const Var& start)const{
      return calcSum(start);
    }
    
    ~MockedActivationFunction(){}
};

#endif
// kate: indent-mode cstyle; space-indent on; indent-width 0; 
