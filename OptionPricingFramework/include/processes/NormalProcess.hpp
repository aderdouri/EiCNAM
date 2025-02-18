#ifndef NORMALPROCESS_HPP
#define NORMALPROCESS_HPP

#include "StochasticProcess.hpp"

class NormalProcess : public StochasticProcess
{
public:
    using StochasticProcess::StochasticProcess;
    torch::Tensor evolve(torch::Tensor S, double dt, torch::Tensor dW) override;
};

#endif // NORMALPROCESS_HPP
