#include "../include/Methods/LongstaffSchwartz.hpp"
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

    int num_steps = 100;
    int num_paths = 500000; // Number of paths to simulate

    // Create LongstaffSchwartz instance
    LongstaffSchwartz longstaffSchwartz(S0, r, sigma, T, num_steps, num_paths);

    // Define exercise times (e.g., every 10 steps)
    std::vector<int> exercise_times;
    for (int i = 10; i < num_steps; i += 10)
    {
        exercise_times.push_back(i);
    }

    // Price the Bermudan option
    torch::Tensor bermudan_option_price = longstaffSchwartz.priceBermudanOption(K, exercise_times);

    // Print the Bermudan option price
    std::cout << "Bermudan Option Price: " << bermudan_option_price.item<double>() << std::endl;

    // Compute first-order gradients (Delta, Rho, Vega, Theta)
    auto grads = torch::autograd::grad({bermudan_option_price}, {S0, r, sigma, T}, /*grad_outputs=*/{}, /*retain_graph=*/true, /*create_graph=*/true);

    auto delta = grads[0];
    auto rho = grads[1];
    auto vega = grads[2];
    auto theta = grads[3];

    std::cout << "Delta (dPrice/dS): " << delta.item<double>() << std::endl;
    std::cout << "Rho (dPrice/dr): " << rho.item<double>() << std::endl;
    std::cout << "Vega (dPrice/dSigma): " << vega.item<double>() << std::endl;
    std::cout << "Theta (dPrice/dT): " << theta.item<double>() << std::endl;

    // Compute Gamma (second derivative with respect to S)
    auto gamma = torch::autograd::grad({delta}, {S0}, /*grad_outputs=*/{}, /*retain_graph=*/false, /*create_graph=*/false)[0];

    std::cout << "Gamma (d2Price/dS2): " << gamma.item<double>() << std::endl;

    return 0;
}
