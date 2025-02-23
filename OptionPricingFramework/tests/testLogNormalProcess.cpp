#include "../include/Processes/LogNormalProcess.hpp"
#include <torch/torch.h>
#include <iostream>

int main()
{
    // Parameters for the Geometric Brownian Motion
    torch::Tensor S0 = torch::tensor(100.0, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor K = torch::tensor(100.0, torch::dtype(torch::kDouble));
    torch::Tensor r = torch::tensor(0.05, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor sigma = torch::tensor(0.2, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor T = torch::tensor(1.0, torch::dtype(torch::kDouble).requires_grad(true));

    int num_steps = 1000;
    int num_paths = 50000; // Number of paths to simulate

    // Create LogNormalProcess instance
    LogNormalProcess logNormalProcess(S0, r, sigma);

    // Simulate the process for num_paths
    torch::Tensor ST = logNormalProcess.simulate(T, num_steps, num_paths);

    // Calculate the payoff for each path
    torch::Tensor payoff = torch::softplus(ST - K);

    // Calculate the average payoff
    torch::Tensor average_payoff = torch::mean(payoff);

    // Discount the average payoff back to present value
    torch::Tensor call_option_price = torch::exp(-r * T) * average_payoff;

    // Compute first-order gradients (Delta, Rho, Vega, Theta)
    auto grads = torch::autograd::grad({call_option_price}, {S0, r, sigma, T}, /*grad_outputs=*/{torch::ones_like(call_option_price)}, /*retain_graph=*/true, /*create_graph=*/true);

    auto delta = grads[0];
    auto rho = grads[1];
    auto vega = grads[2];
    auto theta = grads[3];

    std::cout << "Call Option Price: " << call_option_price.item<double>() << std::endl;
    std::cout << "Delta (dPrice/dS): " << delta.item<double>() << std::endl;
    std::cout << "Rho (dPrice/dr): " << rho.item<double>() << std::endl;
    std::cout << "Vega (dPrice/dSigma): " << vega.item<double>() << std::endl;
    std::cout << "Theta (dPrice/dT): " << -theta.item<double>() << std::endl; // Apply the sign convention here

    // Compute Gamma (second derivative with respect to S)
    auto gamma = torch::autograd::grad({delta}, {S0}, /*grad_outputs=*/{}, /*retain_graph=*/false, /*create_graph=*/false)[0];

    std::cout << "Gamma (d2Price/dS2): " << gamma.item<double>() << std::endl;

    return 0;
}