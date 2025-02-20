#include <torch/torch.h>
#include <iostream>

int main()
{
    // Display PyTorch configuration
    // std::cout << torch::show_config();

    torch::Tensor A = torch::randn({4, 3});
    torch::Tensor Y = torch::randn({4, 1});

    // Normal equation: beta = (A^T A)^(-1) A^T Y
    torch::Tensor AtA = torch::matmul(A.t(), A);
    torch::Tensor AtY = torch::matmul(A.t(), Y);

    // Solve the system
    torch::Tensor beta = torch::matmul(torch::inverse(AtA), AtY);

    std::cout << "Solution beta:\n"
              << beta << std::endl;

    return 0;
}
