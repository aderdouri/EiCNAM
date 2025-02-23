#include <torch/torch.h>
#include <iostream>

namespace
{
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

    // Standard normal probability density function (PDF)
    torch::Tensor normalPDF(torch::Tensor x)
    {
        return torch::exp(-0.5 * x.pow(2)) / std::sqrt(2.0 * M_PI);
    }

    torch::Tensor callPrice(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
    {
        return S * normalCDF(d1(S, K, r, sigma, T)) - K * torch::exp(-r * T) * normalCDF(d2(S, K, r, sigma, T));
    }

    // Closed-form Greeks
    torch::Tensor closedDelta(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
    {
        return normalCDF(d1(S, K, r, sigma, T));
    }

    torch::Tensor closedGamma(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
    {
        return normalPDF(d1(S, K, r, sigma, T)) / (S * sigma * torch::sqrt(T));
    }

    torch::Tensor closedVega(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
    {
        return S * torch::sqrt(T) * normalPDF(d1(S, K, r, sigma, T));
    }

    torch::Tensor closedTheta(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
    {
        torch::Tensor d1_val = d1(S, K, r, sigma, T);
        torch::Tensor d2_val = d2(S, K, r, sigma, T);
        torch::Tensor term1 = -(S * normalPDF(d1_val) * sigma) / (2.0 * torch::sqrt(T));
        torch::Tensor term2 = -r * K * torch::exp(-r * T) * normalCDF(d2_val);
        return term1 + term2;
    }

    torch::Tensor closedRho(torch::Tensor S, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
    {
        torch::Tensor d2_val = d2(S, K, r, sigma, T);
        return K * T * torch::exp(-r * T) * normalCDF(d2_val);
    }
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

    // Closed-form Greeks
    auto delta_closed = closedDelta(S, K, r, sigma, T);
    auto gamma_closed = closedGamma(S, K, r, sigma, T);
    auto vega_closed = closedVega(S, K, r, sigma, T);
    auto theta_closed = closedTheta(S, K, r, sigma, T);
    auto rho_closed = closedRho(S, K, r, sigma, T);

    // Compute first-order gradients (Delta, Rho, Vega, Theta)
    auto grads = torch::autograd::grad({price}, {S, r, sigma, T}, /*grad_outputs=*/{}, /*retain_graph=*/true, /*create_graph=*/true);

    auto delta = grads[0];
    auto rho = grads[1];
    auto vega = grads[2];
    auto theta = -1 * grads[3]; // Sign convention for Theta

    std::cout << "Call Option Price: " << price.item<double>() << std::endl;

    std::cout << "Delta ClosedForm (dPrice/dS): " << delta_closed.item<double>() << std::endl;
    std::cout << "Rho ClosedForm (dPrice/dr): " << rho_closed.item<double>() << std::endl;
    std::cout << "Vega ClosedForm (dPrice/dSigma): " << vega_closed.item<double>() << std::endl;
    std::cout << "Theta ClosedForm (dPrice/dT): " << theta_closed.item<double>() << std::endl;
    std::cout << "Gamma ClosedForm (d2Price/dS2): " << gamma_closed.item<double>() << std::endl;

    std::cout << "==========================================" << std::endl;

    std::cout << "Delta Autograd (dPrice/dS): " << delta.item<double>() << std::endl;
    std::cout << "Rho Autograd (dPrice/dr): " << rho.item<double>() << std::endl;
    std::cout << "Vega Autograd (dPrice/dSigma): " << vega.item<double>() << std::endl;
    std::cout << "Theta Autograd (dPrice/dT): " << theta.item<double>() << std::endl;

    // Compute Gamma (second derivative with respect to S)
    auto gamma = torch::autograd::grad({delta}, {S}, /*grad_outputs=*/{}, /*retain_graph=*/false, /*create_graph=*/false)[0];

    std::cout << "Gamma Autograd (d2Price/dS2): " << gamma.item<double>() << std::endl;

    return 0;
}
