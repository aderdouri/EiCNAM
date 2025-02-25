#ifndef INTENSITYPROCESS_HPP
#define INTENSITYPROCESS_HPP

#include "StochasticProcess.hpp"

class IntensityProcess : public StochasticProcess
{
public:
    IntensityProcess(double mu, double sigma, double k, double nu, const std::string &device = "cpu");
    torch::Tensor evolve(torch::Tensor S, double dt, torch::Tensor dW) override;

private:
    double k;
    double nu;
};

#endif // INTENSITYPROCESS_HPP
