#include "../../include/Methods/NormalProcess.hpp"

torch::Tensor NormalProcess::evolve(torch::Tensor S, double dt, torch::Tensor dW)
{
    return S + mu * dt + sigma * dW;
}
