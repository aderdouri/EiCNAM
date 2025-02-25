#ifndef CIRPLUSPLUSPROCESS_HPP
#define CIRPLUSPLUSPROCESS_HPP

#include "StochasticProcess.hpp"
#include <functional>

class CIRPlusPlusProcess : public StochasticProcess
{
public:
    CIRPlusPlusProcess(double mu, double sigma, double k, double theta, double nu, std::function<double(double)> phi, const std::string &device = "cpu");
    torch::Tensor evolve(torch::Tensor S, double dt, torch::Tensor dW, double t);

private:
    double k;
    double theta;
    double nu;
    std::function<double(double)> phi;
};

#endif // CIRPLUSPLUSPROCESS_HPP
