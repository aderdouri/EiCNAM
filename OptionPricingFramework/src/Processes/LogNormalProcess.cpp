#include "../../include/Processes/LogNormalProcess.hpp"

torch::Tensor LogNormalProcess::evolve(torch::Tensor S, double dt, torch::Tensor dW)
{
    return S * torch::exp((mu - 0.5 * sigma * sigma) * dt + sigma * dW);
}

torch::Tensor LogNormalProcess::simulate(torch::Tensor T, int num_steps, int num_paths)
{
    // Ensure T has requires_grad = true for autograd
    TORCH_CHECK(T.requires_grad(), "T must have requires_grad=true for Theta calculation.");

    // Compute time step size
    torch::Tensor dt = T / num_steps;

    // Expand S0 for all paths and initialize Brownian motion
    torch::Tensor S = S0.expand({num_paths}).clone();
    torch::Tensor dW = torch::randn({num_paths, num_steps}, torch::dtype(torch::kDouble)) * torch::sqrt(dt);

    // Simulate the process over time
    for (int i = 0; i < num_steps; ++i)
    {
        S = evolve(S, dt.item<double>(), dW.select(1, i));
    }

    return S;
}
