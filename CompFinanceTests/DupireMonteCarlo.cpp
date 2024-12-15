//
// Date: December 2024
// File name: DupireMonteCarlo.cpp
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

TEST(DupireMonteCarlo, DupireMonteCarlo01)
{
    std::println("DupireMonteCarlo...");

    try
    {
        std::vector<double> spots = {90.0, 100.0,
                                     110.0, 120.0, 130.0,
                                     140.0, 150.0, 160.0, 170.0, 180.0};

        std::vector<Time> times = {0.1000, 0.5750, 1.0500, 1.5250, 2.0000};
        matrix<double> lVols(10, 5);

        for (int i = 0; i < lVols.rows(); ++i)
        {
            for (int j = 0; j < lVols.cols(); ++j)
            {
                lVols[i][j] = 0.20;
            }
        }

        // Dupire Barrier parameters
        const double S0{100.0};
        const double maturity = 2.0;
        const double strike = 120.0;
        const double barrier = 150.0;
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

        std::cout << "[\n";
        for (int i = 0; i < vegas.rows(); ++i)
        {
            std::cout << "[";
            for (int j = 0; j < vegas.cols(); ++j)
            {
                std::cout << std::setw(12) << std::setprecision(4) << std::fixed << vegas[i][j] << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "]\n";
        // int row = 9;
        // int col = 2;
        // std::println("Vega[{}][{}]: {}", row, col, vegas[row][col]);

        // row = 9;
        // col = 4;
        // std::println("Vega[{}][{}]: {}", row, col, vegas[row][col]);
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
}