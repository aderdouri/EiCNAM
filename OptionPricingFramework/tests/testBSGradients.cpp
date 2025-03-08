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

    // Compute first-order gradients (Delta, Rho, Vega, Theta)
    auto grads = torch::autograd::grad({price}, {S, r, sigma, T}, /*grad_outputs=*/{torch::ones_like(price)}, /*retain_graph=*/true, /*create_graph=*/true);

    auto delta = grads[0];
    auto rho = grads[1];
    auto vega = grads[2];
    auto theta = -1 * grads[3]; // Theta is -dV/dT

    std::cout << "Call Option Price: " << price.item<double>() << std::endl;
    std::cout << "Delta (dPrice/dS): " << delta.item<double>() << std::endl;
    std::cout << "Rho (dPrice/dr): " << rho.item<double>() << std::endl;
    std::cout << "Vega (dPrice/dSigma): " << vega.item<double>() << std::endl;
    std::cout << "Theta (dPrice/dT): " << theta.item<double>() << std::endl;

    // Compute Gamma (second derivative with respect to S)
    auto gamma = torch::autograd::grad({delta}, {S}, /*grad_outputs=*/{torch::ones_like(delta)}, /*retain_graph=*/true, /*create_graph=*/true)[0];

    return 0;
}
