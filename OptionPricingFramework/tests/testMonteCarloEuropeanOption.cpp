#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <random>

namespace
{
    torch::Tensor monteCarloCallPrice(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T, int num_simulations)
    {
        // Generate standard normal random numbers using torch
        torch::Tensor Z = torch::randn({num_simulations}, torch::dtype(torch::kDouble));

        // Compute stock price at maturity ST
        torch::Tensor ST = S * torch::exp((r - 0.5 * sigma * sigma) * T + sigma * torch::sqrt(T) * Z);

        // Compute payoff
        torch::Tensor payoff = torch::softplus(ST - K);

        // Discounted expected payoff
        torch::Tensor call_price = torch::mean(payoff) * torch::exp(-r * T);

        return call_price;
    }
}

int main()
{
    // Enable anomaly detection
    torch::autograd::AnomalyMode::set_enabled(true);

    int num_simulations = 1000000; // Large number for accuracy

    // Ensure all tensors are double precision and require gradients where needed
    torch::Tensor S = torch::tensor(100.0, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor K = torch::tensor(100.0, torch::dtype(torch::kDouble));
    torch::Tensor r = torch::tensor(0.05, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor sigma = torch::tensor(0.2, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor T = torch::tensor(1.0, torch::dtype(torch::kDouble).requires_grad(true));

    torch::Tensor price = monteCarloCallPrice(S, K, r, sigma, T, num_simulations);

    // Compute first-order gradients (Delta, Rho, Vega, Theta)
    auto grads = torch::autograd::grad({price}, {S, r, sigma, T}, /*grad_outputs=*/{}, /*retain_graph=*/true, /*create_graph=*/true);

    auto delta = grads[0];
    auto rho = grads[1];
    auto vega = grads[2];
    auto theta = grads[3];

    std::cout << "Call Option Price: " << price.item<double>() << std::endl;
    std::cout << "Delta (dPrice/dS): " << delta.item<double>() << std::endl;
    std::cout << "Rho (dPrice/dr): " << rho.item<double>() << std::endl;
    std::cout << "Vega (dPrice/dSigma): " << vega.item<double>() << std::endl;
    std::cout << "Theta (dPrice/dT): " << theta.item<double>() << std::endl;

    // Compute Gamma (second derivative with respect to S)
    auto gamma = torch::autograd::grad({delta}, {S}, /*grad_outputs=*/{}, /*retain_graph=*/false, /*create_graph=*/false)[0];

    std::cout << "Gamma (d2Price/dS2): " << gamma.item<double>() << std::endl;

    return 0;
}
