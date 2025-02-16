#ifndef MONTECARLO_HPP
#define MONTECARLO_HPP

#include <torch/torch.h>

torch::Tensor monteCarloCallPrice(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T, int num_simulations);
torch::Tensor monteCarloPutPrice(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T, int num_simulations);

#endif // MONTECARLO_HPP
