#include "BlackFormula.h"

double BlackFormula::price(double forward, double volatility, double numeraire, double strike, double expiry, bool isCall)
{
    double periodVolatility = volatility * std::sqrt(expiry);
    double dPlus = std::log(forward / strike) / periodVolatility + 0.5 * periodVolatility;
    double dMinus = dPlus - periodVolatility;
    double omega = isCall ? 1.0 : -1.0;
    double nPlus = NORMAL.cdf(omega * dPlus);
    double nMinus = NORMAL.cdf(omega * dMinus);
    double price = numeraire * omega * (forward * nPlus - strike * nMinus);
    return price;
}

DoubleDerivatives BlackFormula::price_Sad(double forward, double volatility, double numeraire, double strike, double expiry, bool isCall)
{
    double omega = isCall ? 1.0 : -1.0;
    double periodVolatility = volatility * std::sqrt(expiry);
    double dPlus = std::log(forward / strike) / periodVolatility + 0.5 * periodVolatility;
    double dMinus = dPlus - periodVolatility;
    double nPlus = NORMAL.cdf(omega * dPlus);
    double nMinus = NORMAL.cdf(omega * dMinus);
    double price = numeraire * omega * (forward * nPlus - strike * nMinus);

    int nbInputs = 5;
    std::vector<double> inputDot(nbInputs, 1.0);
    std::vector<double> periodVolatilityDot(nbInputs, 0.0);
    periodVolatilityDot[1] = std::sqrt(expiry) * inputDot[1];
    periodVolatilityDot[4] = volatility * 0.5 / std::sqrt(expiry) * inputDot[4];

    std::vector<double> dPlusDot(nbInputs, 0.0);
    for (int i = 0; i < nbInputs; ++i)
    {
        dPlusDot[i] = (std::log(forward / strike) * -1.0 / (periodVolatility * periodVolatility) + 0.5) * periodVolatilityDot[i];
    }
    dPlusDot[0] += 1.0 / (periodVolatility * forward) * inputDot[0];
    dPlusDot[3] += -1.0 / (periodVolatility * forward) * inputDot[3];

    std::vector<double> dMinusDot(nbInputs, 0.0);
    for (int i = 0; i < nbInputs; ++i)
    {
        dMinusDot[i] = dPlusDot[i] - periodVolatilityDot[i];
    }

    std::vector<double> nPlusDot(nbInputs, 0.0);
    double nPdfpPlus = NORMAL.pdf(omega * dPlus);
    for (int i = 0; i < nbInputs; ++i)
    {
        nPlusDot[i] = nPdfpPlus * omega * dPlusDot[i];
    }

    std::vector<double> nMinusDot(nbInputs, 0.0);
    double nPdfdMinus = NORMAL.pdf(omega * dMinus);
    for (int i = 0; i < nbInputs; ++i)
    {
        nMinusDot[i] = nPdfdMinus * omega * dMinusDot[i];
    }

    std::vector<double> priceDot(nbInputs, 0.0);
    for (int i = 0; i < nbInputs; ++i)
    {
        priceDot[i] = numeraire * omega * forward * nPlusDot[i] - numeraire * omega * strike * nMinusDot[i];
    }
    priceDot[0] += numeraire * omega * nPlus * inputDot[0];
    priceDot[2] += omega * (forward * nPlus - strike * nMinus) * inputDot[2];
    priceDot[3] += -numeraire * omega * nMinus * inputDot[3];

    return DoubleDerivatives(price, priceDot);
}

DoubleDerivatives BlackFormula::price_Aad(double forward, double volatility, double numeraire, double strike, double expiry, bool isCall)
{
    double omega = isCall ? 1.0 : -1.0;
    double periodVolatility = volatility * std::sqrt(expiry);
    double dPlus = std::log(forward / strike) / periodVolatility + 0.5 * periodVolatility;
    double dMinus = dPlus - periodVolatility;
    double nPlus = NORMAL.cdf(omega * dPlus);
    double nMinus = NORMAL.cdf(omega * dMinus);
    double price = numeraire * omega * (forward * nPlus - strike * nMinus);

    double priceBar = 1.0;
    double nMinusBar = numeraire * omega * -strike * priceBar;
    double nPlusBar = numeraire * omega * forward * priceBar;
    double dMinusBar = NORMAL.pdf(omega * dMinus) * omega * nMinusBar;
    double dPlusBar = 1.0 * dMinusBar + NORMAL.pdf(omega * dPlus) * omega * nPlusBar;

    double periodVolatilityBar = -1.0 * dMinusBar + (-std::log(forward / strike) / (periodVolatility * periodVolatility) + 0.5) * dPlusBar;
    std::vector<double> inputBar(5, 0.0);
    inputBar[4] = volatility * 0.5 / std::sqrt(expiry) * periodVolatilityBar;
    inputBar[3] = -1.0 / strike / periodVolatility * dPlusBar + numeraire * omega * -nMinus * priceBar;
    inputBar[2] = omega * (forward * nPlus - strike * nMinus) * priceBar;
    inputBar[1] = std::sqrt(expiry) * periodVolatilityBar;
    inputBar[0] = 1.0 / forward / periodVolatility * dPlusBar + numeraire * omega * nPlus * priceBar;

    return DoubleDerivatives(price, inputBar);
}