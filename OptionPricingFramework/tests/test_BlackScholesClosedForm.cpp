#include <torch/torch.h>
#include <iostream>

torch::Tensor d1(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
{
    return (torch::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * torch::sqrt(T));
}

torch::Tensor d2(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
{
    return d1(S, K, r, sigma, T) - sigma * torch::sqrt(T);
}

torch::Tensor normalCDF(torch::Tensor x)
{
    return 0.5 * torch::erfc(-x * torch::sqrt(torch::tensor(0.5, torch::dtype(torch::kDouble))));
}

torch::Tensor callPrice(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
{
    return S * normalCDF(d1(S, K, r, sigma, T)) - K * torch::exp(-r * T) * normalCDF(d2(S, K, r, sigma, T));
}

int main()
{
    // Ensure all tensors are double precision and require gradients where needed
    torch::Tensor S = torch::tensor(100.0, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor K = torch::tensor(100.0, torch::dtype(torch::kDouble));
    torch::Tensor r = torch::tensor(0.05, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor sigma = torch::tensor(0.2, torch::dtype(torch::kDouble).requires_grad(true));
    torch::Tensor T = torch::tensor(1.0, torch::dtype(torch::kDouble).requires_grad(true));

    torch::Tensor price = callPrice(S, K, r, sigma, T);

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
