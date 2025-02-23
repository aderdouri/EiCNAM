#include <torch/torch.h>
#include <iostream>
#include "../include/Methods/BlackScholesClosedForm.hpp"

int main()
{
    // Ensure all tensors are double precision and require gradients where needed
    torch::Tensor S = torch::tensor(100.0, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor K = torch::tensor(100.0, torch::dtype(torch::kDouble));
    torch::Tensor r = torch::tensor(0.05, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor sigma = torch::tensor(0.2, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor T = torch::tensor(1.0, torch::dtype(torch::kDouble).requires_grad(true));

    // Create BlackScholes instance
    BlackScholes blackScholes(S, K, r, sigma, T);

    // Compute option price and Greeks
    torch::Tensor price = blackScholes.price();
    torch::Tensor delta = blackScholes.delta();
    torch::Tensor gamma = blackScholes.gamma();
    torch::Tensor vega = blackScholes.vega();
    torch::Tensor theta = blackScholes.theta();
    torch::Tensor rho = blackScholes.rho();

    std::cout << "Call Option Price: " << price.item<double>() << std::endl;
    std::cout << "Delta (dPrice/dS): " << delta.item<double>() << std::endl;
    std::cout << "Gamma (d2Price/dS2): " << gamma.item<double>() << std::endl;
    std::cout << "Vega (dPrice/dSigma): " << vega.item<double>() << std::endl;
    std::cout << "Theta (dPrice/dT): " << theta.item<double>() << std::endl;
    std::cout << "Rho (dPrice/dr): " << rho.item<double>() << std::endl;

    return 0;
}
