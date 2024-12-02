#include <iostream>
#include <cmath>
#include <vector>

/**
 * Returns cos(a[0]+ exp(a[1])) * (sin(a[2]) + cos(a[3])) + pow(a[1], 1.5) + a[3].
 * @param a The parameters
 * @return The result.
 */
double f(const std::vector<double> &a)
{
    double b1 = a[0] + std::exp(a[1]);
    double b2 = std::sin(a[2]) + std::cos(a[3]);
    double b3 = std::pow(a[1], 1.5) + a[3];
    double b4 = std::cos(b1) * b2 + b3;
    return b4;
}

/**
 * Returns the value of the function f and its derivatives with respect to all the inputs.
 * The function is f(a) = cos(a[0]+ exp(a[1])) * (sin(a[2]) + cos(a[3])) + pow(a[1], 1.5) + a[3].
 * The derivatives are computed by Standard Algorithmic Differentiation.
 * @param a The parameters
 * @return The value of f and its derivatives.
 */
std::pair<double, std::vector<double>> f_Sad(const std::vector<double> &a)
{
    // Forward sweep - function
    double b1 = a[0] + std::exp(a[1]);
    double b2 = std::sin(a[2]) + std::cos(a[3]);
    double b3 = std::pow(a[1], 1.5) + a[3];
    double b4 = std::cos(b1) * b2 + b3;

    // Forward sweep - derivatives
    int nbA = a.size();
    std::vector<double> b1Dot(nbA, 0.0);
    b1Dot[0] = 1.0;
    b1Dot[1] = std::exp(a[1]);

    std::vector<double> b2Dot(nbA, 0.0);
    b2Dot[2] = std::cos(a[2]);
    b2Dot[3] = -std::sin(a[3]);

    std::vector<double> b3Dot(nbA, 0.0);
    b3Dot[1] = 1.5 * std::sqrt(a[1]);
    b3Dot[3] = 1.0;

    std::vector<double> b4Dot(nbA, 0.0);
    for (int loopa = 0; loopa < nbA; loopa++)
    {
        b4Dot[loopa] = b2 * -std::sin(b1) * b1Dot[loopa] +
                       std::cos(b1) * b2Dot[loopa] + 1.0 * b3Dot[loopa];
    }

    return std::make_pair(b4, b4Dot);
}

int main()
{
    // Test the function f
    std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
    double result = f(a);
    std::cout << "f(a) = " << result << std::endl;

    // Test the function f_Sad
    auto [value, derivatives] = f_Sad(a);
    std::cout << "f(a) = " << value << std::endl;
    for (size_t i = 0; i < derivatives.size(); ++i)
    {
        std::cout << "df/da[" << i << "] = " << derivatives[i] << std::endl;
    }

    return 0;
}