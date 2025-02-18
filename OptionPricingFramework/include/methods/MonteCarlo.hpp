#ifndef MONTECARLO_HPP
#define MONTECARLO_HPP

#include <torch/torch.h>

class MonteCarlo
{
public:
    MonteCarlo(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T, int num_simulations);
    torch::Tensor callPrice();
    torch::Tensor putPrice();

private:
    torch::Tensor S;
    torch::Tensor K;
    torch::Tensor r;
    torch::Tensor sigma;
    torch::Tensor T;
    int num_simulations;
};

#endif // MONTECARLO_HPP
