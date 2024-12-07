//
// Date: December 2024
// File name: BlackScholes.cpp
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
#include "../CppUnitTest/TestHarness.h"
#include "../CompFinance/analytics.h"

TEST(mertonTest, mertonTest01)
{
    std::println ("mertonTest...");

    double spot = 100.0;
    double strike = 100.0;
    double vol = 0.15;
    double mat = 1.0;
    double intens = 0.1;
    double meanJmp = 0.1;
    double stdJmp = 0.1;

    double price = merton(spot, strike, vol, mat, intens, meanJmp, stdJmp   );
    println("price {}: ", price);
}