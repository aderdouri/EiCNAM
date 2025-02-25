#include "../../include/Methods/BS.hpp"

BlackScholes::BlackScholes(double S0, double K, double r, double sigma, double T)
    : S0(S0), K(K), r(r), sigma(sigma), T(T) {}

double BlackScholes::d1()
{
    return (std::log(S0 / K) + (r + 0.5 * std::pow(sigma, 2)) * T) / (sigma * std::sqrt(T));
}

double BlackScholes::d2()
{
    return d1() - sigma * std::sqrt(T);
}

double BlackScholes::normalCDF(double x)
{
    return 0.5 * (1 + std::erf(x / std::sqrt(2.0)));
}

double BlackScholes::normalPDF(double x)
{
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * std::pow(x, 2));
}

double BlackScholes::price()
{
    double D1 = d1();
    double D2 = d2();
    return S0 * normalCDF(D1) - K * std::exp(-r * T) * normalCDF(D2);
}

double BlackScholes::delta()
{
    return normalCDF(d1());
}

double BlackScholes::gamma()
{
    double D1 = d1();
    return normalPDF(D1) / (S0 * sigma * std::sqrt(T));
}

double BlackScholes::vega()
{
    double D1 = d1();
    return S0 * normalPDF(D1) * std::sqrt(T);
}

double BlackScholes::theta()
{
    double D1 = d1();
    double D2 = d2();
    return -(S0 * normalPDF(D1) * sigma) / (2 * std::sqrt(T)) - r * K * std::exp(-r * T) * normalCDF(D2);
}

double BlackScholes::rho()
{
    double D2 = d2();
    return K * T * std::exp(-r * T) * normalCDF(D2);
}
