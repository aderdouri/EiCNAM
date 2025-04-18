#include <torch/torch.h>
#include <torch/script.h> // Include this for torch::linalg
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
using namespace torch;

class LongstaffSchwartz
{
public:
    LongstaffSchwartz(torch::Tensor S0, torch::Tensor r, torch::Tensor sigma, double K, int N, int M, torch::Tensor T)
        : S0(S0), r(r), sigma(sigma), K(K), N(N), M(M), T(T) {}

    Tensor simulatePaths()
    {
        Tensor dt = T / N;
        Tensor Z = torch::randn({M, N}, kFloat32);
        Tensor S = torch::ones({M, N}, kFloat32) * S0;

        for (int t = 1; t < N; ++t)
        {
            Tensor St_minus_1 = S.index({torch::indexing::Slice(), t - 1});
            Tensor St = St_minus_1 * torch::exp((r - 0.5 * sigma * sigma) * dt + sigma * torch::sqrt(dt) * Z.index({torch::indexing::Slice(), t - 1}));
            S = torch::cat({S.index({torch::indexing::Slice(), torch::indexing::Slice(0, t)}), St.unsqueeze(1)}, 1);
        }

        return S;
    }

    torch::Tensor price(const Tensor &paths)
    {
        Tensor dt = T / N;
        Tensor exercise = torch::relu(K - paths);
        Tensor values = exercise.index({torch::indexing::Slice(), N - 1}).clone();

        for (int t = N - 2; t >= 0; --t)
        {
            Tensor in_the_money = exercise.index({torch::indexing::Slice(), t}) > 0;
            Tensor x = paths.index({in_the_money, t});
            Tensor y = values.index({in_the_money}) * torch::exp(-r * dt);

            if (x.size(0) > 0)
            {
                torch::Tensor A = torch::stack({torch::ones_like(x), x, x * x}, 1);
                torch::Tensor AtA = torch::matmul(A.t(), A);
                torch::Tensor AtY = torch::matmul(A.t(), y);
                torch::Tensor beta = torch::matmul(torch::inverse(AtA), AtY);
                torch::Tensor continuation_value = torch::matmul(A, beta).squeeze();

                torch::Tensor exercise_value = K - x;
                torch::Tensor exercise = exercise_value > continuation_value;
                Tensor exercise_indices = torch::nonzero(exercise).squeeze();

                if (exercise_indices.numel() > 0)
                {
                    Tensor new_values = values.clone();
                    new_values.index_put_({exercise_indices}, exercise_value.index({exercise}));
                    values = new_values;
                }
            }
        }

        Tensor price = torch::mean(values * torch::exp(-r * dt));
        return price;
    }

private:
    torch::Tensor S0, r, sigma, T;
    double K;
    int N, M;
};

int main()
{
    // Enable anomaly detection
    torch::autograd::AnomalyMode::set_enabled(true);

    // Parameters
    torch::Tensor S0 = torch::tensor(100.0, torch::dtype(torch::kFloat32).requires_grad(true));
    torch::Tensor r = torch::tensor(0.05, torch::dtype(torch::kFloat32).requires_grad(true));
    torch::Tensor sigma = torch::tensor(0.2, torch::dtype(torch::kFloat32).requires_grad(true));
    double K = 100.0;
    const int N = 100;
    const int M = 500000;
    torch::Tensor T = torch::tensor(1.0, torch::dtype(torch::kFloat32).requires_grad(true));

    // Create LongstaffSchwartz instance
    LongstaffSchwartz longstaffSchwartz(S0, r, sigma, K, N, M, T);

    // Generate paths
    Tensor paths = longstaffSchwartz.simulatePaths();
    Tensor price = longstaffSchwartz.price(paths);

    // Compute first-order gradients (Delta, Rho, Vega, Theta) with create_graph = true
    auto grads = torch::autograd::grad({price}, {S0, r, sigma, T}, /*grad_outputs=*/{torch::ones_like(price)}, /*retain_graph=*/true, /*create_graph=*/true);

    auto delta = grads[0];
    auto rho = grads[1];
    auto vega = grads[2];
    auto theta = grads[3];

    std::cout << "Call Option Price: " << price.item<double>() << std::endl;
    std::cout << "Delta (dPrice/dS): " << delta.item<double>() << std::endl;
    std::cout << "Rho (dPrice/dr): " << rho.item<double>() << std::endl;
    std::cout << "Vega (dPrice/dSigma): " << vega.item<double>() << std::endl;
    std::cout << "Theta (dPrice/dT): " << theta.item<double>() << std::endl;

    // Compute Gamma (second derivative with respect to S) by retaining the graph
    auto gamma = torch::autograd::grad({delta}, {S0}, /*grad_outputs=*/{torch::ones_like(delta)}, /*retain_graph=*/false, /*create_graph=*/false)[0];

    std::cout << "Gamma (d2Price/dS2): " << gamma.item<double>() << std::endl;

    return 0;
}
