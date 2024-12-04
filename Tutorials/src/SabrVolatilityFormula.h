#if !defined(SABR_VOLATILITY_FORMULA_ALREADY_INCLUDED)
#define SABR_VOLATILITY_FORMULA_ALREADY_INCLUDED
#include "Utils.h"

class SabrVolatilityFormula
{
private:
    static constexpr double Z_RANGE = 1.0E-6;

public:
    static double volatility(double forward, double alpha, double beta, double rho,
                             double nu, double strike, double expiry);

    static DoubleDerivatives volatility_Aad(double forward, double alpha, double beta, double rho, double nu, double strike, double expiry);
};

#endif // SABR_VOLATILITY_FORMULA_ALREADY_INCLUDED