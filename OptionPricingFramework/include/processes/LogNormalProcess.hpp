#ifndef LOGNORMALPROCESS_HPP
#define LOGNORMALPROCESS_HPP

#include "StochasticProcess.hpp"
#include <torch/torch.h>

class LogNormalProcess : public StochasticProcess
{
public:
    LogNormalProcess(torch::Tensor S0, torch::Tensor mu, torch::Tensor sigma)
        : StochasticProcess(S0, mu, sigma) {}

    torch::Tensor evolve(torch::Tensor S, double dt, torch::Tensor dW) override;
    torch::Tensor simulate(torch::Tensor T, int num_steps, int num_paths);
};

#endif // LOGNORMALPROCESS_HPP
