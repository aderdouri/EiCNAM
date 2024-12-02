#include <cmath>
#include <vector>
#include <iostream>
#include <stdexcept>

double normal_cdf(double x)
{
    return 0.5 * std::erfc(-x * std::sqrt(0.5));
}

double normal_pdf(double x)
{
    return std::exp(-0.5 * x * x) / std::sqrt(2 * M_PI);
}

double price(double forward, double volatility, double numeraire, double strike, double expiry, bool isCall)
{
    double periodVolatility = volatility * std::sqrt(expiry);
    double dPlus = std::log(forward / strike) / periodVolatility + 0.5 * periodVolatility;
    double dMinus = dPlus - periodVolatility;
    double omega = isCall ? 1.0 : -1.0;
    double nPlus = normal_cdf(omega * dPlus);
    double nMinus = normal_cdf(omega * dMinus);
    double price = numeraire * omega * (forward * nPlus - strike * nMinus);
    return price;
}

std::vector<double> price_Sad(double forward, double volatility, double numeraire, double strike, double expiry, bool isCall)
{
    double omega = isCall ? 1.0 : -1.0;
    double periodVolatility = volatility * std::sqrt(expiry);
    double dPlus = std::log(forward / strike) / periodVolatility + 0.5 * periodVolatility;
    double dMinus = dPlus - periodVolatility;
    double nPlus = normal_cdf(omega * dPlus);
    double nMinus = normal_cdf(omega * dMinus);
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
    double nPdfpPlus = normal_pdf(omega * dPlus);
    for (int i = 0; i < nbInputs; ++i)
    {
        nPlusDot[i] = nPdfpPlus * omega * dPlusDot[i];
    }

    std::vector<double> nMinusDot(nbInputs, 0.0);
    double nPdfdMinus = normal_pdf(omega * dMinus);
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

    std::vector<double> result = {price};
    result.insert(result.end(), priceDot.begin(), priceDot.end());
    return result;
}

int main()
{
    double forward = 100.0;
    double volatility = 0.2;
    double numeraire = 1.0;
    double strike = 100.0;
    double expiry = 1.0;
    bool isCall = true;

    double optionPrice = price(forward, volatility, numeraire, strike, expiry, isCall);
    std::cout << "Option Price: " << optionPrice << std::endl;

    std::vector<double> priceAndDerivatives = price_Sad(forward, volatility, numeraire, strike, expiry, isCall);
    std::cout << "Option Price (with SAD): " << priceAndDerivatives[0] << std::endl;
    std::cout << "Derivatives: ";
    for (size_t i = 1; i < priceAndDerivatives.size(); ++i)
    {
        std::cout << priceAndDerivatives[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}