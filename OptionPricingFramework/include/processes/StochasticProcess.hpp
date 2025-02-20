#ifndef STOCHASTICPROCESS_HPP
#define STOCHASTICPROCESS_HPP

#include <torch/torch.h>
#include <string>

class StochasticProcess
{
public:
    StochasticProcess(torch::Tensor S0, torch::Tensor mu, torch::Tensor sigma)
        : S0(S0), mu(mu), sigma(sigma) {}

    virtual torch::Tensor evolve(torch::Tensor S, double dt, torch::Tensor dW) = 0;

protected:
    torch::Tensor S0;
    torch::Tensor mu;
    torch::Tensor sigma;
};

#endif // STOCHASTICPROCESS_HPP
