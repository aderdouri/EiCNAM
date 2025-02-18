#include "../../include/Methods/CIRPlusPlusProcess.hpp"

CIRPlusPlusProcess::CIRPlusPlusProcess(double mu, double sigma, double k, double theta, double nu, std::function<double(double)> phi, const std::string &device)
    : StochasticProcess(mu, sigma, device), k(k), theta(theta), nu(nu), phi(phi) {}

torch::Tensor CIRPlusPlusProcess::evolve(torch::Tensor S, double dt, torch::Tensor dW, double t)
{
    torch::Tensor drift = k * (mu - S) * dt;
    torch::Tensor diffusion = nu * torch::sqrt(S) * torch::sqrt(dt) * dW;
    torch::Tensor shift = theta * dt + phi(t) * dt;
    return S + drift + diffusion; // + shift;
}
