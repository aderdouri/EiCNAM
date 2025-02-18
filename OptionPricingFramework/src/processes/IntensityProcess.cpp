#include "../../include/Methods/IntensityProcess.hpp"

IntensityProcess::IntensityProcess(double mu, double sigma, double k, double nu, const std::string &device)
    : StochasticProcess(mu, sigma, device), k(k), nu(nu) {}

torch::Tensor IntensityProcess::evolve(torch::Tensor S, double dt, torch::Tensor dW)
{
    torch::Tensor drift = k * (mu - S) * dt;
    torch::Tensor diffusion = nu * torch::sqrt(S) * torch::sqrt(dt) * dW;
    return S + drift + diffusion;
}
