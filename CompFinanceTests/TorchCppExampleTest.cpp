#include <torch/torch.h>
#include <iostream>

// Function: f(x) = x^2 + 3x + 5
torch::Tensor function(torch::Tensor x)
{
    return x.pow(2) + 3 * x + 5;
}

int main()
{
    torch::Tensor x = torch::tensor(2.0, torch::requires_grad());

    // Compute function value
    torch::Tensor y = function(x);

    // Compute gradients
    y.backward();

    // Print function value and gradient
    std::cout << "Function value: " << y.item<double>() << std::endl;
    std::cout << "Gradient (dy/dx): " << x.grad().item<double>() << std::endl;

    return 0;
}
