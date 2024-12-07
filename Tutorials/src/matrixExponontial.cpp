#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using namespace Eigen;

// Define a template function to compute the square of a value
template <typename T>
T square(const T &x)
{
    return x * x;
}

// Define the function that computes the matrix B
template <typename T>
Matrix<T, Dynamic, Dynamic> computeMatrix(const Matrix<T, Dynamic, Dynamic> &A)
{
    // Extract elements of A
    T a = A(0, 0), b = A(0, 1), c = A(1, 0), d = A(1, 1), delta;

    // Compute delta
    delta = std::sqrt(square(a - d) + 4 * b * c);

    // Initialize matrix B
    Matrix<T, Dynamic, Dynamic> B(2, 2);

    // Fill matrix B
    B(0, 0) = std::exp(0.5 * (a + d)) * (delta * std::cosh(0.5 * delta) + (a - d) * std::sinh(0.5 * delta));
    B(0, 1) = 2 * b * std::exp(0.5 * (a + d)) * std::sinh(0.5 * delta);
    B(1, 0) = 2 * c * std::exp(0.5 * (a + d)) * std::sinh(0.5 * delta);
    B(1, 1) = std::exp(0.5 * (a + d)) * (delta * std::cosh(0.5 * delta) + (d - a) * std::sinh(0.5 * delta));

    // Return the normalized matrix B
    return B / delta;
}

int main()
{
    // Example usage
    Matrix<double, Dynamic, Dynamic> A(2, 2);
    A << 1.0, 2.0,
        3.0, 4.0;

    Matrix<double, Dynamic, Dynamic> B = computeMatrix(A);

    // Print the result
    std::cout << "Matrix B:\n"
              << B << std::endl;

    return 0;
}
