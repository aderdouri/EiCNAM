#include <iostream>
#include <cmath>
#include <vector>

struct DoubleDerivatives
{
    double value;
    std::vector<double> derivatives;

    DoubleDerivatives(double val, std::vector<double> derivs) : value(val), derivatives(derivs) {}
};

double f(const std::vector<double> &a)
{
    double b1 = a[0] + std::exp(a[1]);
    double b2 = std::sin(a[2]) + std::cos(a[3]);
    double b3 = std::pow(a[1], 1.5) + a[3];
    double b4 = std::cos(b1) * b2 + b3;
    return b4;
}

DoubleDerivatives f_Aad(const std::vector<double> &a)
{
    // Forward sweep - function
    double b1 = a[0] + std::exp(a[1]);
    double b2 = std::sin(a[2]) + std::cos(a[3]);
    double b3 = std::pow(a[1], 1.5) + a[3];
    double b4 = std::cos(b1) * b2 + b3;

    // Backward sweep - derivatives
    std::vector<double> aBar(a.size(), 0.0);
    double b4Bar = 1.0;
    double b3Bar = 1.0 * b4Bar;
    double b2Bar = std::cos(b1) * b4Bar;
    double b1Bar = b2 * -std::sin(b1) * b4Bar;
    aBar[3] = 1.0 * b3Bar - std::sin(a[3]) * b2Bar;
    aBar[2] = std::cos(a[2]) * b2Bar;
    aBar[1] = 1.5 * std::sqrt(a[1]) * b3Bar + std::exp(a[1]) * b1Bar;
    aBar[0] = 1.0 * b1Bar;

    return DoubleDerivatives(b4, aBar);
}

int main()
{
    std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
    double result = f(a);
    DoubleDerivatives resultAad = f_Aad(a);

    std::cout << "Function value: " << result << std::endl;
    std::cout << "Function value (AAD): " << resultAad.value << std::endl;
    std::cout << "Derivatives: ";
    for (double derivative : resultAad.derivatives)
    {
        std::cout << derivative << " ";
    }
    std::cout << std::endl;

    return 0;
}