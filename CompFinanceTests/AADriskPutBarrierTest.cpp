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
#include "../CppUnitTest/TestHarness.h"
#include "../CompFinance/main.h"

TEST(AADriskPutBarrierTest, AADriskPutBarrierTest01)
{
    return;
    std::cout << "AADriskPutBarrierTest...\n";

    try
    {
        // Product details
        const std::string productid{"ko"};
        std::string riskPayoff{};

        double strike = 120.0;
        Time maturity = 3.0;
        double barrier = 150.0;
        double smoothing = 0.02;
        double barried_dt = 1.0 / 52.0;
        bool callPut = false;

        // Initialize put barrier product
        putBarrier(strike, barrier, maturity, barried_dt, smoothing, callPut, productid);

        // Black-Scholes model parameters
        double spot = 100.0;
        double vol = 0.15;
        bool qSpot = false;
        double rate = 0.02;
        double div = 0.03;
        const std::string modelid = "BS";
        putBlackScholes(spot, vol, qSpot, rate, div, modelid);

        // Numerical parameters for simulation
        bool parallel{true};
        bool useSobol{true};
        int numPath{500000};
        int seed1{12345};
        int seed2{12346};

        NumericalParam num{parallel, useSobol, numPath, seed1, seed2};

        // Perform AAD risk analysis
        auto results = AADriskOne(modelid, productid, num, riskPayoff);
        std::cout << "results.riskPayoffValue: " << results.riskPayoffValue << '\n';

        // Output individual risks
        const size_t n = results.risks.size(), N = n + 1;
        for (const auto &risk : results.risks)
        {
            std::cout << "risk: " << risk << '\n';
        }
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
}