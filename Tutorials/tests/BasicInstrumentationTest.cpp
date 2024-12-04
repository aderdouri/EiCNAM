//
// Date: December 2024
// File name: BasicInstrumentationTest.cpp
// Version: 1
// Author: Abderrazak DERDOURI
// Subject:
//
// Description:
//
//
// Notes:
// Revision History:
//

#include <iostream>
#include <print>
#include "../../CompFinance/AAD.h"
#include "../../CppUnitTest/TestHarness.h"

namespace
{
    template <class T>
    T f(T x[5])
    {
        auto y1 = x[2] * (5.0 * x[0] + x[1]);
        auto y2 = log(y1);
        auto y = (y1 + x[3] * y2) * (y1 + y2);
        return y;
    }

    template <class T>
    T fStarter(const std::vector<T> &a)
    {
        T b1 = a[0] + exp(a[1]);
        T b2 = sin(a[2]) + cos(a[3]);
        T b3 = pow(a[1], 1.5) + a[3];
        T b4 = cos(b1) * b2 + b3;
        return b4;
    }
}

TEST(BasicInstrumentation, BasicInstrumentationTest01)
{
    std::println("BasicInstrumentationTest");
    Number::tape->rewind();

    Number x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    x->putOnTape();

    // Evaluate and build the tape
    Number y = f(x);

    // Propagate adjoints through the tape in reverse order
    y.propagateToStart();

    // Get derivatives
    // 950.736, 190.147, 443.677, 73.2041, 0
    for (size_t i = 0; i < 5; ++i)
    {
        std::println("a: {}, = {}", i, x[i].adjoint());
    }
}

TEST(BasicInstrumentation, BasicInstrumentationTest02)
{

    std::println("BasicInstrumentationTest02");
    Number::tape->rewind();

    std::vector<Number> x{1.0, 2.0, 3.0, 4.0};
    for (auto &num : x)
    {
        num.putOnTape();
    }

    // Evaluate and build the tape
    Number y = fStarter(x);

    // Propagate adjoints through the tape in reverse order
    y.propagateToStart();

    // Get derivatives
    // 0.440889 5.37907 0.504802 0.614103
    for (size_t i = 0; i < x.size(); ++i) 
    {
        std::println("a: {}, = {}", i, x[i].adjoint());
    }
}
