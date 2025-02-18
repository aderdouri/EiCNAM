#include "../../include/Methods/LogNormalProcess.hpp"

torch::Tensor LogNormalProcess::evolve(torch::Tensor S, double dt, torch::Tensor dW)
{
    return S * torch::exp((mu - 0.5 * sigma * sigma) * dt + sigma * dW);
}
