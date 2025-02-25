#include "../../include/Methods/LongstaffSchwartz.hpp"
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <algorithm>

LongstaffSchwartz::LongstaffSchwartz(torch::Tensor S0, torch::Tensor r, torch::Tensor sigma, torch::Tensor T, int num_steps, int num_paths)
    : S0(S0), r(r), sigma(sigma), T(T), num_steps(num_steps), num_paths(num_paths) {}

torch::Tensor LongstaffSchwartz::simulatePaths()
{
    torch::Tensor dt = T / num_steps;
    torch::Tensor paths = torch::zeros({num_paths, num_steps + 1}, torch::dtype(torch::kDouble));
    paths.index_put_({torch::indexing::Slice(), 0}, S0);

    for (int t = 1; t <= num_steps; ++t)
    {
        torch::Tensor Z = torch::randn({num_paths}, torch::dtype(torch::kDouble));
        paths.index_put_({torch::indexing::Slice(), t},
                         paths.index({torch::indexing::Slice(), t - 1}) * torch::exp((r - 0.5 * sigma * sigma) * dt + sigma * torch::sqrt(dt) * Z));
    }

    return paths;
}

torch::Tensor LongstaffSchwartz::priceBermudanOption(torch::Tensor K, const std::vector<int> &exercise_times)
{
    torch::Tensor paths = simulatePaths();
    torch::Tensor dt = T / num_steps;
    torch::Tensor discount_factor = torch::exp(-r * dt);

    torch::Tensor payoff = torch::max(paths.index({torch::indexing::Slice(), -1}) - K, torch::zeros({num_paths}, torch::dtype(torch::kDouble)));
    torch::Tensor continuation_value = torch::zeros_like(payoff);

    for (int t = num_steps - 1; t >= 0; --t)
    {
        if (std::find(exercise_times.begin(), exercise_times.end(), t) != exercise_times.end())
        {
            torch::Tensor in_the_money = (paths.index({torch::indexing::Slice(), t}) > K);
            torch::Tensor X = paths.index({torch::indexing::Slice(), t}).masked_select(in_the_money);
            torch::Tensor Y = payoff.masked_select(in_the_money).unsqueeze(1); // Ensure Y is a column vector [n, 1]

            if (X.size(0) > 0)
            {
                // Create design matrix A with [1, X, X^2] for least squares regression
                torch::Tensor A = torch::stack({torch::ones_like(X), X, X * X}, 1); // Shape [n, 3]

                // Compute normal equations: beta = (A^T A)^(-1) A^T Y
                torch::Tensor AtA = torch::matmul(A.t(), A); // Shape [3, 3]
                torch::Tensor AtY = torch::matmul(A.t(), Y); // Shape [3, 1]

                // Solve the system using inverse (for simplicity)
                torch::Tensor beta = torch::matmul(torch::inverse(AtA), AtY); // Shape [3, 1]

                // Calculate continuation value
                continuation_value = torch::matmul(A, beta).squeeze(); // Shape [n]
            }

            // Create continuation value tensor with full size and fill with zeros
            torch::Tensor full_continuation = torch::zeros({num_paths}, torch::dtype(torch::kDouble));

            // Assign values only for in-the-money paths
            full_continuation = full_continuation.masked_scatter(in_the_money, continuation_value);

            // Compare exercise and continuation values
            torch::Tensor exercise_value = torch::max(paths.index({torch::indexing::Slice(), t}) - K, torch::zeros({num_paths}, torch::dtype(torch::kDouble)));
            payoff = torch::max(exercise_value, full_continuation * discount_factor);
        }
        else
        {
            payoff = payoff * discount_factor;
        }
    }

    // Calculate the final option price
    return torch::mean(payoff) * torch::exp(-r * dt);
}
