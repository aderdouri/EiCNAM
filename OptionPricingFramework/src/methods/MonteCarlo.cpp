#include "../../include/Methods/MonteCarlo.hpp"
#include <cmath>
#include <random>

MonteCarlo::MonteCarlo(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T, int num_simulations)
    : S(S), K(K), r(r), sigma(sigma), T(T), num_simulations(num_simulations) {}

torch::Tensor MonteCarlo::callPrice()
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

torch::Tensor MonteCarlo::putPrice()
{
    // Generate standard normal random numbers using torch
    torch::Tensor Z = torch::randn({num_simulations}, torch::dtype(torch::kDouble));

    // Compute stock price at maturity ST
    torch::Tensor ST = S * torch::exp((r - 0.5 * sigma * sigma) * T + sigma * torch::sqrt(T) * Z);

    // Compute payoff
    torch::Tensor payoff = torch::softplus(K - ST);

    // Discounted expected payoff
    torch::Tensor put_price = torch::mean(payoff) * torch::exp(-r * T);

    return put_price;
}
