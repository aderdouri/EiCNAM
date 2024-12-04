#if !defined(UTILS_H_ALREADY_INCLUDED)
#define UTILS_H_ALREADY_INCLUDED

#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <random>

class Normal
{
public:
    Normal(double mean, double stddev);
    double cdf(double x);
    double pdf(double x);

private:
    std::normal_distribution<double> dist;
};

class DoubleDerivatives
{
public:
    DoubleDerivatives(double value, const std::vector<double> &derivatives);

    double getValue() const;
    const std::vector<double> &getDerivatives() const;

private:
    double value;
    std::vector<double> derivatives;
};

#endif // UTILS_H_ALREADY_INCLUDED