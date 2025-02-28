#include <torch/torch.h>
#include <iostream>

int main() {
    // Check if Metal backend is available
    if (torch::hasMPS()) {
        std::cout << "Metal (MPS) is available! Running on GPU.\n";
    } else {
        std::cout << "Metal (MPS) is NOT available. Running on CPU.\n";
    }

    // Create a tensor on the GPU (MPS backend)
    torch::Device device(torch::kMPS);  // Use Metal GPU

    // Create a random tensor on GPU
    torch::Tensor tensor = torch::rand({3, 3}, device);

    // Perform a simple computation
    torch::Tensor result = tensor * 2 + 1;

    std::cout << "Input Tensor (GPU):\n" << tensor << "\n";
    std::cout << "Result Tensor (GPU):\n" << result << "\n";

    return 0;
}
