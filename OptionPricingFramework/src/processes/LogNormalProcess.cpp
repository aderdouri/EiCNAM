#include "../../include/Processes/LogNormalProcess.hpp"

torch::Tensor LogNormalProcess::evolve(torch::Tensor S, double dt, torch::Tensor dW)
{
    return S * torch::exp((mu - 0.5 * sigma * sigma) * dt + sigma * dW);
}

torch::Tensor LogNormalProcess::simulate(torch::Tensor T, int num_steps, int num_paths)
{
    double dt = T.item<double>() / num_steps;
    torch::Tensor S = S0.expand({num_paths});
    torch::Tensor dW = torch::randn({num_paths, num_steps}, torch::dtype(torch::kDouble)) * std::sqrt(dt);

    for (int i = 0; i < num_steps; ++i)
    {
        S = evolve(S, dt, dW.select(1, i));
    }

    return S;
}
