#include <iostream>
#include <cmath>
#include <array>
#include <vector>

#include "Utils.h"
#include "BlackFormula.h"
#include "SabrVolatilityFormula.h"

class SabrPriceFormula
{
public:
  static double price(
      double forward,
      double alpha,
      double beta,
      double rho,
      double nu,
      double numeraire,
      double strike,
      double expiry,
      bool isCall)
  {
    double volatility = SabrVolatilityFormula::volatility(forward, alpha, beta, rho, nu, strike, expiry);
    double price = BlackFormula::price(forward, volatility, numeraire, strike, expiry, isCall);
    return price;
  }

  static DoubleDerivatives price_Aad(
      double forward,
      double alpha,
      double beta,
      double rho,
      double nu,
      double numeraire,
      double strike,
      double expiry,
      bool isCall)
  {
    DoubleDerivatives volatility = SabrVolatilityFormula::volatility_Aad(forward, alpha, beta, rho, nu, strike, expiry);
    DoubleDerivatives price = BlackFormula::price_Aad(forward, volatility.getValue(), numeraire, strike, expiry, isCall);
    double priceBar = 1.0;
    double volatilityBar = price.getDerivatives()[1];
    std::vector<double> inputBar(8, 0.0); // forward, alpha, beta, rho, nu, numeraire, strike, expiry
    inputBar[7] += price.getDerivatives()[4] * priceBar;
    inputBar[7] += volatility.getDerivatives()[6] * volatilityBar;
    inputBar[6] += price.getDerivatives()[3] * priceBar;
    inputBar[6] += volatility.getDerivatives()[5] * volatilityBar;
    inputBar[5] += price.getDerivatives()[2] * priceBar;
    inputBar[4] += volatility.getDerivatives()[4] * volatilityBar;
    inputBar[3] += volatility.getDerivatives()[3] * volatilityBar;
    inputBar[2] += volatility.getDerivatives()[2] * volatilityBar;
    inputBar[1] += volatility.getDerivatives()[1] * volatilityBar;
    inputBar[0] += price.getDerivatives()[0] * priceBar;
    inputBar[0] += volatility.getDerivatives()[0] * volatilityBar;
    return DoubleDerivatives(price.getValue(), inputBar);
  }
};

int main()
{
  double forward = 100.0;
  double alpha = 0.2;
  double beta = 0.5;
  double rho = -0.3;
  double nu = 0.4;
  double numeraire = 1.0;
  double strike = 100.0;
  double expiry = 1.0;
  bool isCall = true;

  double optionPrice = SabrPriceFormula::price(forward, alpha, beta, rho, nu, numeraire, strike, expiry, isCall);
  std::cout << "Option Price: " << optionPrice << std::endl;

  DoubleDerivatives optionPriceAad = SabrPriceFormula::price_Aad(forward, alpha, beta, rho, nu, numeraire, strike, expiry, isCall);
  std::cout << "Option Price (AAD): " << optionPriceAad.getValue() << std::endl;
  std::cout << "Derivatives: ";
  for (double derivative : optionPriceAad.getDerivatives())
  {
    std::cout << derivative << " ";
  }
  std::cout << std::endl;

  return 0;
}
