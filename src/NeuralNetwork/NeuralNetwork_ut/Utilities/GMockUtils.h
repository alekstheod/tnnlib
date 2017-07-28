#ifndef GMOCKUTILS_H
#define GMOCKUTILS_H

#include <gmock/gmock.h>

/**
 * @brief forward declaration of the supportTest function.
 */
#define USING_SUPPORT_TEST_T(FIXTURE, TESTNAME, CLASSNAME) \
    class FIXTURE##_##TESTNAME##_Test;                     \
    template <> template <> void ::CLASSNAME::supportTest (FIXTURE##_##TESTNAME##_Test& test);

#define USING_SUPPORT_TEST(FIXTURE, TESTNAME, CLASSNAME) \
    class FIXTURE##_##TESTNAME##_Test;                   \
    template <> void ::CLASSNAME::supportTest (FIXTURE##_##TESTNAME##_Test& test);

/**
 * @brief forward declaration of the supportTest function in a namespace NAMESPACE
 */
#define USING_SUPPORT_TEST_N(FIXTURE, TESTNAME, NAMESPACE, CLASSNAME)                              \
    class FIXTURE##_##TESTNAME##_Test;                                                             \
    namespace NAMESPACE {                                                                          \
        template <> template <> void ::CLASSNAME::supportTest (FIXTURE##_##TESTNAME##_Test& test); \
    }

#define USING_SUPPORT_TEST_T_N(FIXTURE, TESTNAME, NAMESPACE, CLASSNAME)                            \
    class FIXTURE##_##TESTNAME##_Test;                                                             \
    namespace NAMESPACE {                                                                          \
        template <> template <> void ::CLASSNAME::supportTest (FIXTURE##_##TESTNAME##_Test& test); \
    }

#define USING_SUPPORT_TEST_T_NN(FIXTURE, TESTNAME, NAMESPACE, NESTED_NAMESPACE, CLASSNAME)             \
    class FIXTURE##_##TESTNAME##_Test;                                                                 \
    namespace NAMESPACE {                                                                              \
        namespace NESTED_NAMESPACE {                                                                   \
            template <> template <> void ::CLASSNAME::supportTest (FIXTURE##_##TESTNAME##_Test& test); \
        }                                                                                              \
    }

/**
 * @brief specialization of the template function supportTest.
 */
#define SUPPORT_TEST_T(FIXTURE, TESTNAME, CLASSNAME) template <> template <> void CLASSNAME::supportTest (FIXTURE##_##TESTNAME##_Test& test)

#define SUPPORT_TEST(FIXTURE, TESTNAME, CLASSNAME) template <> void CLASSNAME::supportTest (FIXTURE##_##TESTNAME##_Test& test)

#endif