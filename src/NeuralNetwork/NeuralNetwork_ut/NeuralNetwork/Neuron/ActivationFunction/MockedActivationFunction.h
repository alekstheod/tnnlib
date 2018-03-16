#ifndef EQUATIONMOCK_H
#define EQUATIONMOCK_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnull-dereference"
#include <gmock/gmock.h>
#pragma clang diagnostic pop

#include <vector>


template< typename VarType >
class MockedActivationFunction {
  public:
    typedef VarType Var;

    template< typename NewVarType >
    using use = MockedActivationFunction< NewVarType >;

  public:
    MockedActivationFunction() {
    }
    MOCK_CONST_METHOD0_T(calcEquation, Var());
    MOCK_CONST_METHOD1_T(calculateDerivate, Var(const Var& input));
    MOCK_CONST_METHOD1_T(calcSum, Var(const Var& start));

    template< typename It >
    Var sum(It begin, It end, const Var& start) const {
        return calcSum(start);
    }

    template< typename It >
    Var calculate(const Var& sum, It begin, It end) const {
        return calcEquation();
    }
    ~MockedActivationFunction() {
    }
};

#endif
