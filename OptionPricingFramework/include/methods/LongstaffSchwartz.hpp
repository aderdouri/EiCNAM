#ifndef LONGSTAFFSCHWARTZ_H
#define LONGSTAFFSCHWARTZ_H

#include <torch/torch.h>
#include <torch/script.h> // Include this for torch::linalg
#include <vector>
#include <iostream>
#include <cmath>

class LongstaffSchwartz
{
public:
    LongstaffSchwartz(torch::Tensor S0, torch::Tensor r, torch::Tensor sigma, double K, int N, int M, torch::Tensor T);

    torch::Tensor simulatePaths();

    torch::Tensor price(const torch::Tensor &paths);

private:
    torch::Tensor S0, r, sigma, T;
    double K;
    int N, M;
};

#endif // LONGSTAFFSCHWARTZ_H
