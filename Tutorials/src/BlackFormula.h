#if !defined(BLACK_FORMULA_H_ALREADY_INCLUDED)
#define BLACK_FORMULA_H_ALREADY_INCLUDED

#include <vector>
#include "Utils.h"

class BlackFormula
{
public:
    static double price(double forward, double volatility, double numeraire, double strike, double expiry, bool isCall);
    static DoubleDerivatives price_Sad(double forward, double volatility, double numeraire, double strike, double expiry, bool isCall);
    static DoubleDerivatives price_Aad(double forward, double volatility, double numeraire, double strike, double expiry, bool isCall);

private:
    static inline Normal NORMAL = Normal(0.0, 1.0);
};
    
#endif // BLACK_FORMULA_H_ALREADY_INCLUDED