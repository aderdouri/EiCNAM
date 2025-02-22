#include "../../include/Methods/LongstaffSchwartz.hpp"
#include <torch/torch.h>
#include <torch/script.h> // Include this for torch::linalg
#include <vector>
#include <iostream>
#include <cmath>

LongstaffSchwartz::LongstaffSchwartz(torch::Tensor S0, torch::Tensor r, torch::Tensor sigma, double K, int N, int M, torch::Tensor T)
    : S0(S0), r(r), sigma(sigma), K(K), N(N), M(M), T(T) {}

torch::Tensor LongstaffSchwartz::simulatePaths()
{
    torch::Tensor dt = T / N;
    torch::Tensor Z = torch::randn({M, N}, torch::kFloat32);
    torch::Tensor S = torch::ones({M, N}, torch::kFloat32) * S0;

    for (int t = 1; t < N; ++t)
    {
        torch::Tensor St_minus_1 = S.index({torch::indexing::Slice(), t - 1});
        torch::Tensor St = St_minus_1 * torch::exp((r - 0.5 * sigma * sigma) * dt + sigma * torch::sqrt(dt) * Z.index({torch::indexing::Slice(), t - 1}));
        S = torch::cat({S.index({torch::indexing::Slice(), torch::indexing::Slice(0, t)}), St.unsqueeze(1)}, 1);
    }

    return S;
}

torch::Tensor LongstaffSchwartz::price(const torch::Tensor &paths)
{
    torch::Tensor dt = T / N;
    torch::Tensor exercise = torch::relu(K - paths);
    torch::Tensor values = exercise.index({torch::indexing::Slice(), N - 1}).clone();

    for (int t = N - 2; t >= 0; --t)
    {
        torch::Tensor in_the_money = exercise.index({torch::indexing::Slice(), t}) > 0;
        torch::Tensor x = paths.index({in_the_money, t});
        torch::Tensor y = values.index({in_the_money}) * torch::exp(-r * dt);

        if (x.size(0) > 0)
        {
            torch::Tensor A = torch::stack({torch::ones_like(x), x, x * x}, 1);
            torch::Tensor AtA = torch::matmul(A.t(), A);
            torch::Tensor AtY = torch::matmul(A.t(), y);
            torch::Tensor beta = torch::matmul(torch::inverse(AtA), AtY);
            torch::Tensor continuation_value = torch::matmul(A, beta).squeeze();

            torch::Tensor exercise_value = K - x;
            torch::Tensor exercise = exercise_value > continuation_value;
            torch::Tensor exercise_indices = torch::nonzero(exercise).squeeze();

            if (exercise_indices.numel() > 0)
            {
                torch::Tensor new_values = values.clone();
                new_values.index_put_({exercise_indices}, exercise_value.index({exercise}));
                values = new_values;
            }
        }
    }

    torch::Tensor price = torch::mean(values * torch::exp(-r * dt));
    return price;
}