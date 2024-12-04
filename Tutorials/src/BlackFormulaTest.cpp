#include "BlackFormula.h"

int main()
{
    double forward = 100.0;
    double volatility = 0.2;
    double numeraire = 1.0;
    double strike = 100.0;
    double expiry = 1.0;
    bool isCall = true;

    double price = BlackFormula::price(forward, volatility, numeraire, strike, expiry, isCall);
    std::cout << "Price: " << price << std::endl;

    DoubleDerivatives sadResult = BlackFormula::price_Sad(forward, volatility, numeraire, strike, expiry, isCall);
    std::cout << "Price (SAD): " << sadResult.getValue() << std::endl;
    for (double derivative : sadResult.getDerivatives())
    {
        std::cout << "Derivative: " << derivative << std::endl;
    }

    DoubleDerivatives aadResult = BlackFormula::price_Aad(forward, volatility, numeraire, strike, expiry, isCall);
    std::cout << "Price (AAD): " << aadResult.getValue() << std::endl;
    for (double derivative : aadResult.getDerivatives())
    {
        std::cout << "Derivative: " << derivative << std::endl;
    }

    return 0;
}