#include "../../include/Methods/BinomialTree.hpp"
#include <torch/torch.h>
#include <cmath>

BinomialTree::BinomialTree(torch::Tensor S0, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T, int N)
    : S0(S0), K(K), r(r), sigma(sigma), T(T), N(N) {}

torch::Tensor BinomialTree::price()
{
    return priceOption();
}

torch::Tensor BinomialTree::priceOption()
{
    torch::Tensor dt = T / N;
    torch::Tensor u = torch::exp(sigma * torch::sqrt(dt));
    torch::Tensor d = torch::exp(-sigma * torch::sqrt(dt));
    torch::Tensor p = (torch::exp(r * dt) - d) / (u - d);

    // Ensure dtype and device consistency
    torch::Tensor optionPrice = torch::zeros({N + 1}, S0.options());

    // Terminal payoff at maturity
    for (int i = 0; i <= N; ++i)
    {
        torch::Tensor stockPrice = S0 * torch::pow(u, N - i) * torch::pow(d, i);
        optionPrice[i] = torch::max(stockPrice - K, torch::tensor(0.0, S0.options()));
    }

    // Backward induction
    for (int j = N - 1; j >= 0; --j)
    {
        torch::Tensor newOptionPrice = torch::zeros({j + 1}, S0.options());
        for (int i = 0; i <= j; ++i)
        {
            newOptionPrice[i] = (p * optionPrice[i] + (1 - p) * optionPrice[i + 1]) * torch::exp(-r * dt);
        }
        optionPrice = newOptionPrice;
    }

    return optionPrice[0];
}
