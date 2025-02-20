#ifndef LONGSTAFFSCHWARTZ_HPP
#define LONGSTAFFSCHWARTZ_HPP

#include <torch/torch.h>
#include <vector>

class LongstaffSchwartz
{
public:
    LongstaffSchwartz(torch::Tensor S0, torch::Tensor r, torch::Tensor sigma, torch::Tensor T, int num_steps, int num_paths);
    torch::Tensor priceBermudanOption(torch::Tensor K, const std::vector<int> &exercise_times);

private:
    torch::Tensor S0;
    torch::Tensor r;
    torch::Tensor sigma;
    torch::Tensor T;
    int num_steps;
    int num_paths;

    torch::Tensor simulatePaths();
};

#endif // LONGSTAFFSCHWARTZ_HPP