//
// Date: December 2024
// File name: AADriskPutBarrierTest.cpp
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
#include <string>
#include <vector>
#include <exception>
#include <print>
#include "../CppUnitTest/TestHarness.h"
#include "../CompFinance/main.h"


TEST(AADriskPutBarrierTest, AADriskPutBarrierTest01)
{
    std::println("AADriskPutBarrierTest...");

    try
    {
        // Product
        const std::string productid{"ko"};
        std::string riskPayoff{};

        double strike = 120.0;
        Time maturity = 3.0;
        double barrier = 150.0;
        double smoothing = 0.02;
        double barried_dt = 1.0 / 52.0;
        bool callPut = false;

        // call 3.00 120.00 up and out 150.00 monitoring freq 0.02 smooth 0.02
        putBarrier(strike, barrier, maturity, barried_dt, smoothing, callPut, productid);

        // Black-Scholes model
        double spot = 100.0;
        double vol = 0.15;
        bool qSpot = false;
        double rate = 0.02;
        double div = 0.03;
        const std::string modelid = "BS";
        putBlackScholes(spot, vol, qSpot, rate, div, modelid);

        // Numerical parameters
        bool parallel{true};
        bool useSobol{true};
        int numPath{500000};
        int seed1{12345};
        int seed2{12346};

        NumericalParam num{parallel, useSobol, numPath, seed1, seed2};

        auto results = AADriskOne(modelid, productid, num, riskPayoff);
        println("results.riskPayoffValue: {}", results.riskPayoffValue);

        const size_t n = results.risks.size(), N = n + 1;
        for (const auto &risk : results.risks)
        {
            println("risk: {}", risk);
        }
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
}