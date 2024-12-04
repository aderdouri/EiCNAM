#include "Utils.h"

Normal::Normal(double mean, double stddev) : dist(mean, stddev) {}

double Normal::cdf(double x)
{
    return 0.5 * std::erfc(-x / std::sqrt(2));
}

double Normal::pdf(double x)
{
    return std::exp(-0.5 * x * x) / std::sqrt(2 * M_PI);
}

DoubleDerivatives::DoubleDerivatives(double value, const std::vector<double> &derivatives)
    : value(value), derivatives(derivatives) {}

double DoubleDerivatives::getValue() const { return value; }

const std::vector<double> &DoubleDerivatives::getDerivatives() const { return derivatives; }
