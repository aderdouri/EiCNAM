//
// Date: December 2024
// File name: BasicInstrumentationExprTest.cpp
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
}

TEST(BasicInstrumentationExpr, BasicInstrumentationExprTest)
{
	std::println("BasicInstrumentationExprTest");

    Number::tape->rewind();

	Number x[5] = { 1.0, 2.0, 3.0, 4.0, 5.0 };

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
