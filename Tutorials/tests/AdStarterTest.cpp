//
// Date: December 2024
// File name: gradientDescentTest.cpp
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
#include "../src/AdStarter.h"
#include "../../CppUnitTest/TestHarness.h"

TEST(AdStarterTest, AdStarterTest01)
{
    std::vector<double> a = {1.0, 2.0, 3.0, 4.0};

    AdStarter adStarter;
    double result = AdStarter::f(a);
    std::cout << "f(a) = " << result << std::endl;

    DoubleDerivatives sadResult = adStarter.f_Sad(a);
    std::cout << "f_Sad(a) = " << sadResult.getValue() << std::endl;
    std::cout << "Derivatives (Sad): ";
    for (double d : sadResult.getDerivatives())
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    DoubleDerivatives aadResult = AdStarter::f_Aad(a);
    std::cout << "f_Aad(a) = " << aadResult.getValue() << std::endl;
    std::cout << "Derivatives (Aad): ";
    for (double d : aadResult.getDerivatives())
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}