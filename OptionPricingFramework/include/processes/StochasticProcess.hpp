#ifndef STOCHASTICPROCESS_HPP
#define STOCHASTICPROCESS_HPP

#include <torch/torch.h>

class StochasticProcess
{
public:
    StochasticProcess(double mu, double sigma, const std::string &device = "cpu");
    virtual torch::Tensor evolve(torch::Tensor S, double dt, torch::Tensor dW) = 0;

protected:
    double mu;
    double sigma;
    std::string device;
};

#endif // STOCHASTICPROCESS_HPP
