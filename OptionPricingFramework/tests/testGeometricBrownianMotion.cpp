#include "../include/processes/GeometricBrownianMotion.hpp"
#include <torch/torch.h>
#include <iostream>

int main()
{
    // Parameters for the Geometric Brownian Motion
    torch::Tensor S0 = torch::tensor(100.0, torch::dtype(torch::kDouble));
    torch::Tensor mu = torch::tensor(0.05, torch::dtype(torch::kDouble));
    torch::Tensor sigma = torch::tensor(0.2, torch::dtype(torch::kDouble));
    torch::Tensor T = torch::tensor(1.0, torch::dtype(torch::kDouble));
    int num_steps = 1000;

    // Create GeometricBrownianMotion instance
    GeometricBrownianMotion gbm(S0, mu, sigma);

    // Simulate the process
    torch::Tensor S = gbm.simulate(T, num_steps);

    // Print the simulated stock price at maturity
    std::cout << "Simulated Stock Price at Maturity: " << S.item<double>() << std::endl;

    return 0;
}
