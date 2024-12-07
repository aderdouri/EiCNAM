//
// Date: December 2024
// File name: AADriskTest.cpp
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
//#include "../CompFinance/analytics.h"
#include "../CompFinance/main.h"
//#include "../CompFinance/store.h"

namespace 
{
    const auto getPayoffIds(std::string const &id)
    {
        const auto *prd = getProduct<double>(id);
        return prd->payoffLabels();
    };
}

TEST(AADriskTest, AADriskTest01)
{
    std::println("AADriskTest...");

    double spot = 100.0;
    double vol = 0.15;
    bool qSpot = false;
    double rate = 0.02;
    double div = 0.03;
    //const std::string store = "BS";
    
    const Time exerciseDate = 3.0;
    const Time settlementDate = 5;
    const double strike = 110;
    const string store = "call";

    putEuropean(strike, exerciseDate, settlementDate, store);

    //const auto* prd = getProduct<double>("call");

    const std::string modelid{"BS"};
    const std::string productid{"call"};
    const std::string xRiskPayoff;

    // numerical parameters
    bool parallel{true};
    bool useSobol{true};
    int numPath{500000};
    int seed1{12345};
    int seed2{12346};

    NumericalParam num {parallel, useSobol, numPath, seed1, seed2};

    const std::string pid = productid;
    const std::string mid = modelid;

    auto riskPayoff = getPayoffIds(pid);
    std::cout << riskPayoff[0] << std::endl;
    println("PayoffId: {}", riskPayoff[0]);

    try
    {
        auto results = AADriskOne(mid, pid, num, riskPayoff[0]);
        const size_t n = results.risks.size(), N = n + 1;
        for (const auto &risk : results.risks)
        {
            println("risk: {}", risk);
        }
    }
    catch (const std::exception & e) 
    {
        std::cout << e.what() << std::endl;
    }
}