#include "../../include/Methods/BlackScholesClosedForm.hpp"

BlackScholes::BlackScholes(torch::Tensor S0, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
    : S0(S0), K(K), r(r), sigma(sigma), T(T) {}

torch::Tensor BlackScholes::d1()
{
    return (torch::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * torch::sqrt(T));
}

torch::Tensor BlackScholes::d2()
{
    return d1() - sigma * torch::sqrt(T);
}

torch::Tensor BlackScholes::normalCDF(torch::Tensor x)
{
    return 0.5 * torch::erfc(-x * torch::sqrt(torch::tensor(0.5, torch::dtype(torch::kDouble))));
}

torch::Tensor BlackScholes::normalPDF(torch::Tensor x)
{
    return torch::exp(-0.5 * x.pow(2)) / std::sqrt(2.0 * M_PI);
}

torch::Tensor BlackScholes::price()
{
    torch::Tensor d1_val = d1();
    torch::Tensor d2_val = d2();
    torch::Tensor call_price = S0 * normalCDF(d1_val) - K * torch::exp(-r * T) * normalCDF(d2_val);
    return call_price;
}

torch::Tensor BlackScholes::delta()
{
    return normalCDF(d1());
}

torch::Tensor BlackScholes::gamma()
{
    return normalPDF(d1()) / (S0 * sigma * torch::sqrt(T));
}

torch::Tensor BlackScholes::vega()
{
    return S0 * torch::sqrt(T) * normalPDF(d1());
}

torch::Tensor BlackScholes::theta()
{
    torch::Tensor d1_val = d1();
    torch::Tensor d2_val = d2();
    torch::Tensor term1 = -(S0 * normalPDF(d1_val) * sigma) / (2.0 * torch::sqrt(T));
    torch::Tensor term2 = -r * K * torch::exp(-r * T) * normalCDF(d2_val);
    return term1 + term2;
}

torch::Tensor BlackScholes::rho()
{
    torch::Tensor d2_val = d2();
    return K * T * torch::exp(-r * T) * normalCDF(d2_val);
}
