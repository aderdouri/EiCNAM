//
// Date: December 2024
// File name: DupireMonteCarloFDM.cpp
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
#include <chrono>
#include "../CompFinance/toyCode.h"
#include "../CppUnitTest/TestHarness.h"
#include "../CompFinance/main.h"

TEST(DupireMonteCarloFDM, DupireMonteCarloFDM01)
{
    std::println("DupireMonteCarloFDM...");

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
        const int Np = 500000;
        const int Nt = 156;
        const double epsilon = 0.05;
        std::unique_ptr<RNG> rng = std::make_unique<Sobol>();
        rng->init(Nt);

        // Results: value and dV/dS, dV/d(local vols)
        double price;
        double delta;
        matrix<double> vegas(lVols.rows(), lVols.cols());
        for (int i = 0; i < vegas.rows(); ++i)
        {
            for (int j = 0; j < vegas.cols(); ++j)
            {
                vegas[i][j] = 0;
            }
        }

        // double h = 0.1;
        // double perturbed_price_up = toyDupireBarrierMc(
        //     S0 + h,
        //     spots,
        //     times,
        //     lVols,
        //     maturity,
        //     strike,
        //     barrier,
        //     Np,
        //     Nt,
        //     100 * epsilon,
        //     *rng);
        // std::println("perturbed_price_up: {}", perturbed_price_up);

        // double perturbed_price_down = toyDupireBarrierMc(
        //     S0 - h,
        //     spots,
        //     times,
        //     lVols,
        //     maturity,
        //     strike,
        //     barrier,
        //     Np,
        //     Nt,
        //     100 * epsilon,
        //     *rng);
        // std::println("perturbed_price_down: {}", perturbed_price_down);

        // // Finite difference approximation of Delta
        // double delta_fdm = (perturbed_price_up - perturbed_price_down) / (2.0 * h);
        // std::println("delta_fdm: {}", delta_fdm);

        auto start_time = std::chrono::high_resolution_clock::now();
        const double VOL_SHIFT = 0.05;

        matrix<double> prices_plus_vol(lVols.rows(), lVols.cols());
        matrix<double> prices_minus_vol(lVols.rows(), lVols.cols());

        auto perturbed_vols_up = lVols;
        auto perturbed_vols_down = lVols;

        for (size_t i = 0; i < lVols.rows(); ++i)
        {
            for (size_t j = 0; j < lVols.cols(); ++j)
            {
                perturbed_vols_up[i][j] += VOL_SHIFT;
                perturbed_vols_down[i][j] -= VOL_SHIFT;

                prices_plus_vol[i][j] = toyDupireBarrierMc(S0, spots, times, perturbed_vols_up, maturity, strike, barrier, Np, Nt, 100 * epsilon, *rng);
                prices_minus_vol[i][j] = toyDupireBarrierMc(S0, spots, times, perturbed_vols_down, maturity, strike, barrier, Np, Nt, 100 * epsilon, *rng);

                vegas[i][j] = (prices_plus_vol[i][j] - prices_minus_vol[i][j]) / (2 * VOL_SHIFT);

                // reset perturbed_vols
                perturbed_vols_up[i][j] -= VOL_SHIFT;
                perturbed_vols_down[i][j] += VOL_SHIFT;

                std::println("vegas[{}][{}]: {}", i, j, vegas[i][j]);
            }
        }

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

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> execution_time = end_time - start_time;
        std::cout << "Test execution time: " << execution_time.count() << " seconds" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
}