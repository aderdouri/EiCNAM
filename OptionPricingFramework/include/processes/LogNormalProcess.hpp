#ifndef LOGNORMALPROCESS_HPP
#define LOGNORMALPROCESS_HPP

#include "StochasticProcess.hpp"

class LogNormalProcess : public StochasticProcess
{
public:
    using StochasticProcess::StochasticProcess;
    torch::Tensor evolve(torch::Tensor S, double dt, torch::Tensor dW) override;
};

#endif // LOGNORMALPROCESS_HPP
