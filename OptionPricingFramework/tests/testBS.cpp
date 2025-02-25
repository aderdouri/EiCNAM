#include <iostream>
#include "../include/Methods/BS.hpp"

int main()
{
    double S = 100.0;
    double K = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    double T = 1.0;

    // Create BlackScholes instance
    BlackScholes blackScholes(S, K, r, sigma, T);

    // Compute option price and Greeks
    double price = blackScholes.price();
    double delta = blackScholes.delta();
    double gamma = blackScholes.gamma();
    double vega = blackScholes.vega();
    double theta = blackScholes.theta();
    double rho = blackScholes.rho();

    std::cout << "Call Option Price: " << price << std::endl;
    std::cout << "Delta (dPrice/dS): " << delta << std::endl;
    std::cout << "Gamma (d2Price/dS2): " << gamma << std::endl;
    std::cout << "Vega (dPrice/dSigma): " << vega << std::endl;
    std::cout << "Theta (dPrice/dT): " << theta << std::endl;
    std::cout << "Rho (dPrice/dr): " << rho << std::endl;

    return 0;
}
