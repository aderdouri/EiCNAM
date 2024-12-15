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
#include <fstream>
#include <vector>
#include <string>
#include <exception>
#include <print>
#include "../CompFinance/toyCode.h"
#include "../CppUnitTest/TestHarness.h"
#include "../CompFinance/main.h"

namespace
{
    auto dupireCalibInfo()
    {

        // Merton market parameters
        const double spot = 100.0;
        const double vol = 0.15;
        const double jmpIntens = 0.05;
        const double jmpAverage = -0.15;
        const double jmpStd = 0.10;

        // The local vol grid
        // The spots to include
        const std::vector<double> vspots{50.0, 200.0};
        // Maximum space between spots
        const double maxDs = 5.0;
        // The times to include, note NOT 0
        const std::vector<Time> vtimes{5.0};
        // Maximum space between times
        const double maxDt = 5.0 / 60.0;

        auto results = dupireCalib(vspots, maxDs, vtimes, maxDt, spot, vol,
                                   jmpIntens, jmpAverage, jmpStd);
        return results;
    }

    void writeMatrixToFile(const std::string &filename, const matrix<double> &matrix)
    {
        // Open file in write mode
        std::ofstream outFile(filename);

        // Check if file is open
        if (!outFile.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }

        // Write matrix to file
        for (int i = 0; i < matrix.rows(); ++i)
        {
            outFile << "[";
            for (int j = 0; j < matrix.cols(); ++j)
            {
                outFile << matrix[i][j];
                if (j != matrix.cols() - 1)
                {
                    outFile << ", "; // Add comma after each element
                }
            }
            outFile << "],\n"; // New line after each row
        }

        // Close file
        outFile.close();

        std::cout << "Matrix written to file: " << filename << std::endl;
    }

}

TEST(DupireBarrierMcRisksTest, DupireBarrierMcRisksTest01)
{
    return;
    std::println("DupireBarrierMcRisksTest...");

    try
    {
        // Dupire Barrier parameters
        const double S0{100.0};
        auto results = dupireCalibInfo();
        std::vector<double> spots = results.spots;
        std::vector<Time> times = results.times;
        matrix<double> lVols = results.lVols;

        // File name
        std::string filename = "lVols.txt";

        // Write matrix to file
        writeMatrixToFile(filename, lVols);

        const double maturity = 5.0;
        const double strike = 120.0;
        const double barrier = 1500.0;
        const int Np = 100000;
        const int Nt = 156;
        const double epsilon = 0.05;
        std::unique_ptr<RNG> rng = std::make_unique<Sobol>();
        rng->init(Nt);

        // Results: value and dV/dS, dV/d(local vols)
        double price;
        double delta;
        matrix<double> vegas(lVols.rows(), lVols.cols());

        toyDupireBarrierMcRisks(
            S0, spots, times, lVols, maturity, strike, barrier, Np, Nt, 100 * epsilon, *rng, price, delta, vegas);

        std::println("Price {}: ", price);
        std::println("Delta {}: ", delta);

        // for (int i = 0; i < vegas.rows(); ++i)
        // {
        //     for (int j = 0; j < vegas.cols(); ++j)
        //     {
        //         std::println("Vega[{}][{}]: {}", i, j, vegas[i][j]);
        //     }
        // }
        int row = 9;
        int col = 2;
        std::println("Vega[{}][{}]: {}", row, col, vegas[row][col]);

        row = 9;
        col = 4;
        std::println("Vega[{}][{}]: {}", row, col, vegas[row][col]);
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
}