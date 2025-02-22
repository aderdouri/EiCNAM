#include "../include/Methods/LongstaffSchwartz.hpp"
#include <torch/torch.h>
#include <iostream>

// Function to compute first-order gradients
void computeFirstOrderGradients(const torch::Tensor &bermudan_option_price, const torch::Tensor &S0, const torch::Tensor &r, const torch::Tensor &sigma, const torch::Tensor &T)
{
    auto grads = torch::autograd::grad({bermudan_option_price}, {S0, r, sigma, T}, /*grad_outputs=*/{}, /*retain_graph=*/true, /*create_graph=*/true);

    auto delta = grads[0];
    auto rho = grads[1];
    auto vega = grads[2];
    auto theta = grads[3];

    std::cout << "Delta (dPrice/dS): " << delta.item<double>() << std::endl;
    std::cout << "Rho (dPrice/dr): " << rho.item<double>() << std::endl;
    std::cout << "Vega (dPrice/dSigma): " << vega.item<double>() << std::endl;
    std::cout << "Theta (dPrice/dT): " << theta.item<double>() << std::endl;
}

// Function to compute Gamma (second derivative with respect to S)
void computeGamma(const torch::Tensor &delta, const torch::Tensor &S0)
{
    auto gamma = torch::autograd::grad({delta}, {S0}, /*grad_outputs=*/{}, /*retain_graph=*/false, /*create_graph=*/false)[0];
    std::cout << "Gamma (d2Price/dS2): " << gamma.item<double>() << std::endl;
}

// Function to set up parameters for the Geometric Brownian Motion
void setupParameters(torch::Tensor &S0, torch::Tensor &K, torch::Tensor &r, torch::Tensor &sigma, torch::Tensor &T)
{
    S0 = torch::tensor(100.0, torch::dtype(torch::kDouble).requires_grad(true));
    K = torch::tensor(100.0, torch::dtype(torch::kDouble));
    r = torch::tensor(0.05, torch::dtype(torch::kDouble).requires_grad(true));
    sigma = torch::tensor(0.2, torch::dtype(torch::kDouble).requires_grad(true));
    T = torch::tensor(1.0, torch::dtype(torch::kDouble).requires_grad(true));
}

// Function to price the Bermudan option
torch::Tensor priceBermudanOption(LongstaffSchwartz &longstaffSchwartz, const torch::Tensor &K, int num_steps)
{
    std::vector<int> exercise_times;
    for (int i = 10; i < num_steps; i += 10)
    {
        exercise_times.push_back(i);
    }
    return longstaffSchwartz.priceBermudanOption(K, exercise_times);
}

int main()
{
    // Enable anomaly detection
    torch::autograd::AnomalyMode::set_enabled(true);

    // Parameters for the Geometric Brownian Motion
    torch::Tensor S0, K, r, sigma, T;
    setupParameters(S0, K, r, sigma, T);

    int num_steps = 100;
    int num_paths = 500000; // Number of paths to simulate

    // Create LongstaffSchwartz instance
    LongstaffSchwartz longstaffSchwartz(S0, r, sigma, T, num_steps, num_paths);

    // Price the Bermudan option
    torch::Tensor bermudan_option_price = priceBermudanOption(longstaffSchwartz, K, num_steps);

    // Print the Bermudan option price
    std::cout << "Bermudan Option Price: " << bermudan_option_price.item<double>() << std::endl;

    // Ensure all tensors have requires_grad set to true
    S0.set_requires_grad(true);
    r.set_requires_grad(true);
    sigma.set_requires_grad(true);
    T.set_requires_grad(true);

    // Compute first-order gradients (Delta, Rho, Vega, Theta)
    computeFirstOrderGradients(bermudan_option_price, S0, r, sigma, T);

    // Compute Gamma (second derivative with respect to S)
    auto delta = torch::autograd::grad({bermudan_option_price}, {S0}, /*grad_outputs=*/{}, /*retain_graph=*/true, /*create_graph=*/true)[0];
    computeGamma(delta, S0);

    return 0;
}
