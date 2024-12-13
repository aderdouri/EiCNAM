//
// Date: December 2024
// File name: AADriskPutEuropeanTest.cpp
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



TEST(AADriskPutEuropeanTest, AADriskPutEuropeanTest01)
{
    std::println("AADriskPutEuropeanTest...");

    try
    {
        // Product
        const Time exerciseDate = 3.0;
        const Time settlementDate = 5.0;
        const double strike = 110.0;
        const std::string productid{"call"};
        putEuropean(strike, exerciseDate, settlementDate, productid);
        const Product<double> *prd = getProduct<double>(productid);
        std::string riskPayoff = prd->payoffLabels()[0];

        // Black-Scholes model
        double spot = 100.0;
        double vol = 0.15;
        bool qSpot = false;
        double rate = 0.02;
        double div = 0.03;
        const std::string modelid = "BS";
        putBlackScholes(spot, vol, qSpot, rate, div, modelid);

        // Numerical parameters
        bool parallel{false};
        bool useSobol{true};
        int numPath{500000};
        int seed1{12345};
        int seed2{12346};

        NumericalParam num{parallel, useSobol, numPath, seed1, seed2};

        auto results = AADriskOne(modelid, productid, num, riskPayoff);
        println("results.riskPayoffValue: {}", results.riskPayoffValue);

        const std::size_t n = results.risks.size();
        const std::size_t N = n + 1;
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