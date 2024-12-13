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
#include "../CompFinance/main.h"

TEST(BlackScholes, BlackScholesTest)
{
    std::println("BlackScholesTest...");

    double spot = 100.0;
    double vol = 0.15;
    bool qSpot = false;
    double rate = 0.02;
    double div = 0.03;
    const std::string store = "BS";

    putBlackScholes(spot, vol, qSpot, rate, div, store);

    const Model<double> *mdl = getModel<double>(store);
    const BlackScholes<double> *bs = dynamic_cast<const BlackScholes<double> *>(mdl);
    println("spot {}: ", bs->spot());
    println("spot {}: ", bs->vol());
    println("spot {}: ", bs->rate());
    println("spot {}: ", bs->div());
}