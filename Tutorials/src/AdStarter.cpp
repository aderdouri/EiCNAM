#include "AdStarter.h"

double AdStarter::f(const std::vector<double> &a)
{
    double b1 = a[0] + std::exp(a[1]);
    double b2 = std::sin(a[2]) + std::cos(a[3]);
    double b3 = std::pow(a[1], 1.5) + a[3];
    double b4 = std::cos(b1) * b2 + b3;
    return b4;
}

DoubleDerivatives AdStarter::f_Sad(const std::vector<double> &a)
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
    for (int loopa = 0; loopa < nbA; ++loopa)
    {
        b4Dot[loopa] = b2 * -std::sin(b1) * b1Dot[loopa] + std::cos(b1) * b2Dot[loopa] + b3Dot[loopa];
    }

    return DoubleDerivatives(b4, b4Dot);
}

DoubleDerivatives AdStarter::f_Aad(const std::vector<double> &a)
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