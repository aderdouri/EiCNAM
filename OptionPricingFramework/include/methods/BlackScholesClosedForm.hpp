#ifndef BLACK_SCHOLES_CLOSED_FORM_HPP
#define BLACK_SCHOLES_CLOSED_FORM_HPP

#include <torch/torch.h>
#include <cmath>

class BlackScholes
{
public:
    BlackScholes(torch::Tensor S0, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T);

    torch::Tensor price();
    torch::Tensor delta();
    torch::Tensor gamma();
    torch::Tensor vega();
    torch::Tensor theta();
    torch::Tensor rho();

private:
    torch::Tensor d1();
    torch::Tensor d2();
    torch::Tensor normalCDF(torch::Tensor x);
    torch::Tensor normalPDF(torch::Tensor x);

    torch::Tensor S0, K, r, sigma, T;
};

#endif // BLACK_SCHOLES_CLOSED_FORM_HPP
