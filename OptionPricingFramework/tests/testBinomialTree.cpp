#include "../include/Methods/BinomialTree.hpp"
#include <iostream>

int main()
{
    // Enable anomaly detection
    torch::autograd::AnomalyMode::set_enabled(true);

    // Parameters
    torch::Tensor S0 = torch::tensor(100.0, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor K = torch::tensor(100.0, torch::dtype(torch::kDouble));
    torch::Tensor r = torch::tensor(0.05, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor sigma = torch::tensor(0.2, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor T = torch::tensor(1.0, torch::dtype(torch::kDouble).requires_grad(true));
    int N = 100;

    // Create BinomialTree instance
    BinomialTree binomialTree(S0, K, r, sigma, T, N);
    torch::Tensor price = binomialTree.price();

    // Compute first-order gradients (Delta, Rho, Vega, Theta) with create_graph = true
    auto grads = torch::autograd::grad({price}, {S0, r, sigma, T}, /*grad_outputs=*/{torch::ones_like(price)}, /*retain_graph=*/true, /*create_graph=*/true);

    auto delta = grads[0];
    auto rho = grads[1];
    auto vega = grads[2];
    auto theta = grads[3];

    std::cout << "Call Option Price: " << price.item<double>() << std::endl;
    std::cout << "Delta (dPrice/dS): " << delta.item<double>() << std::endl;
    std::cout << "Rho (dPrice/dr): " << rho.item<double>() << std::endl;
    std::cout << "Vega (dPrice/dSigma): " << vega.item<double>() << std::endl;
    std::cout << "Theta (dPrice/dT): " << theta.item<double>() << std::endl;

    // Compute Gamma (second derivative with respect to S) by retaining the graph
    auto gamma = torch::autograd::grad({delta}, {S0}, /*grad_outputs=*/{torch::ones_like(delta)}, /*retain_graph=*/false, /*create_graph=*/false)[0];

    std::cout << "Gamma (d2Price/dS2): " << gamma.item<double>() << std::endl;

    return 0;
}
