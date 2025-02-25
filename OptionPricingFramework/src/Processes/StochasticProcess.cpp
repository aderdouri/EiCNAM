#include "../../include/Processes/StochasticProcess.hpp"

StochasticProcess::StochasticProcess(torch::Tensor S0, torch::Tensor mu, torch::Tensor sigma)
    : S0(S0), mu(mu), sigma(sigma) {}
