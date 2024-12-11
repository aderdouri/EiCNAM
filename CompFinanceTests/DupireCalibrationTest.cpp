//
// Date: December 2024
// File name: DupireCalibrationTest.cpp
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

TEST(DupireCalibrationTest, DupireCalibrationTest01)
{
    std::println("DupireCalibrationTest...");

    try
    {
        // Merton market parameters
        const double spot = 100.0;
        const double vol = 0.15;
        const double jmpIntens = 0.05;
        const double jmpAverage = -0.15;
        const double jmpStd = 0.10;

        // The local vol grid
        // The spots to include
        const std::vector<double> vspots{50.0, 100.0, 200.0};
        // Maximum space between spots
        const double maxDs = 5.0;
        // The times to include, note NOT 0
        const std::vector<Time> vtimes{5.0};
        // Maximum space between times
        const double maxDt = 5.0 / 60.0;

        auto results = dupireCalib(vspots, maxDs, vtimes, maxDt, spot, vol,
                                   jmpIntens, jmpAverage, jmpStd);

        std::vector<double> spots = results.spots;
        std::vector<Time> times = results.times;
        matrix<double> lVols = results.lVols;

        for (auto spot : spots)
        {
            std::cout << spot << std::endl;
        }

        for (auto time : times)
        {
            std::cout << time << std::endl;
        }

        std::size_t rows = lVols.rows();
        std::size_t cols = lVols.cols();
        // std::vector<double>   vect = lVols.getVector(rows, cols);

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                std::cout << lVols[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
}