#ifndef BS_HPP
#define BS_HPP

#include <cmath>

class BlackScholes
{
public:
    BlackScholes(double S0, double K, double r, double sigma, double T);

    double price();
    double delta();
    double gamma();
    double vega();
    double theta();
    double rho();

private:
    double d1();
    double d2();
    double normalCDF(double x);
    double normalPDF(double x);

    double S0, K, r, sigma, T;
};

#endif // BS_HPP
