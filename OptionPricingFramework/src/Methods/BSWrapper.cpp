#include "../../include/Methods/BSWrapper.hpp"

BlackScholesWrapper::BlackScholesWrapper(torch::Tensor S0, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T)
    : S0(S0), K(K), r(r), sigma(sigma), T(T), bs(S0.item<double>(), K.item<double>(), r.item<double>(), sigma.item<double>(), T.item<double>()) {}

torch::Tensor BlackScholesWrapper::price()
{
    return torch::tensor(bs.price(), torch::dtype(torch::kDouble).requires_grad(true));
}

torch::Tensor BlackScholesWrapper::delta()
{
    return torch::autograd::grad({price()}, {S0})[0];
}

torch::Tensor BlackScholesWrapper::gamma()
{
    return torch::autograd::grad({delta()}, {S0})[0];
}

torch::Tensor BlackScholesWrapper::vega()
{
    return torch::autograd::grad({price()}, {sigma})[0];
}

torch::Tensor BlackScholesWrapper::theta()
{
    return torch::autograd::grad({price()}, {T})[0];
}

torch::Tensor BlackScholesWrapper::rho()
{
    return torch::autograd::grad({price()}, {r})[0];
}
