#include "SabrVolatilityFormula.h"

int main()
{
    double forward = 0.03;
    double alpha = 0.2;
    double beta = 0.5;
    double rho = -0.3;
    double nu = 0.4;
    double strike = 0.03;
    double expiry = 1.0;

    double vol = SabrVolatilityFormula::volatility(forward, alpha, beta, rho, nu, strike, expiry);
    std::cout << "Volatility: " << vol << std::endl;

    DoubleDerivatives volAad = SabrVolatilityFormula::volatility_Aad(forward, alpha, beta, rho, nu, strike, expiry);
    std::cout << "Volatility (AAD): " << volAad.getValue() << std::endl;
    std::cout << "Derivatives: ";
    for (double deriv : volAad.getDerivatives())
    {
        std::cout << deriv << " ";
    }
    std::cout << std::endl;

    return 0;
}