#ifndef BS_WRAPPER_HPP
#define BS_WRAPPER_HPP

#include "testBSWrapper.hpp"
#include <torch/torch.h>

class BlackScholesWrapper
{
public:
    BlackScholesWrapper(torch::Tensor S0, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T);

    torch::Tensor price();
    torch::Tensor delta();
    torch::Tensor gamma();
    torch::Tensor vega();
    torch::Tensor theta();
    torch::Tensor rho();

private:
    torch::Tensor S0, K, r, sigma, T;
    BlackScholes bs;
};

#endif // BS_WRAPPER_HPP
