#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <random>

namespace
{
    torch::Tensor monteCarloCallPrice(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T, int num_simulations, int num_steps)
    {
        // Compute time step size (dt)
        torch::Tensor dt = T / num_steps;

        // Initialize stock price paths
        torch::Tensor ST = S.expand({num_simulations});

        // Generate random normal variables for all steps at once
        torch::Tensor Z = torch::randn({num_simulations, num_steps}, torch::dtype(torch::kDouble));

        // Simulate the process over time
        for (int i = 0; i < num_steps; ++i)
        {
            // Update stock price
            ST = ST * torch::exp((r - 0.5 * sigma * sigma) * dt + sigma * torch::sqrt(dt) * Z.select(1, i));
        }

        // Compute correct call option payoff
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

    int num_simulations = 100000; // Large number for accuracy
    int num_steps = 365;          // Number of time steps

    // Ensure all tensors are double precision and require gradients where needed
    torch::Tensor S = torch::tensor(100.0, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor K = torch::tensor(100.0, torch::dtype(torch::kDouble));
    torch::Tensor r = torch::tensor(0.05, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor sigma = torch::tensor(0.2, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor T = torch::tensor(1.0, torch::dtype(torch::kDouble).requires_grad(true));

    torch::Tensor price = monteCarloCallPrice(S, K, r, sigma, T, num_simulations, num_steps);

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

    auto charm = torch::autograd::grad({delta}, {T}, {torch::ones_like(delta)}, true, true)[0];
    auto vanna = torch::autograd::grad({delta}, {sigma}, {torch::ones_like(delta)}, true, true)[0];
    auto vomma = torch::autograd::grad({vega}, {sigma}, {torch::ones_like(vega)}, true, true)[0];
    auto speed = torch::autograd::grad({gamma}, {S}, {torch::ones_like(gamma)}, true, true)[0];
    auto zomma = torch::autograd::grad({gamma}, {sigma}, {torch::ones_like(gamma)}, true, true)[0];
    auto color = torch::autograd::grad({gamma}, {T}, {torch::ones_like(gamma)}, true, true)[0];
    auto dvega_dtime = torch::autograd::grad({vega}, {T}, {torch::ones_like(vega)}, true, true)[0];

    std::cout << "Gamma (d2Price/dS2): " << gamma.item<double>() << std::endl;

    // Elasticity (Lambda)
    auto Lambda = (delta.item<double>() * S.item<double>()) / price.item<double>();

    std::cout << "Lambda (Elasticity):" << Lambda << std::endl;
    std::cout << "Charm (dDelta/dT):" << charm.item<double>() << std::endl;
    std::cout << "Vanna (dDelta/dSigma):" << vanna.item<double>() << std::endl;
    std::cout << "Vomma (dVega/dSigma):" << vomma.item<double>() << std::endl;
    std::cout << "Speed (dGamma/dS):" << speed.item<double>() << std::endl;
    std::cout << "Zomma (dGamma/dSigma):" << zomma.item<double>() << std::endl;
    std::cout << "Color (dGamma/dT):" << color.item<double>() << std::endl;
    std::cout << "DvegaDtime (dVega/dT):" << dvega_dtime.item<double>() << std::endl;

    return 0;
}
