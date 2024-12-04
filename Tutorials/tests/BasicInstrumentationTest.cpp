//
// Date: December 2024
// File name: BasicInstrumentationTest.cpp
// Version: 1
// Author: Abderrazak DERDOURI
// Subject:
//
// Description:
//
//
// Notes:
// Revision History:
//

#include <iostream>
#include <print>
#include "../../CompFinance/AAD.h"
#include "../../CppUnitTest/TestHarness.h"

namespace
{
    template <class T>
    T f(T x[5])
    {
        auto y1 = x[2] * (5.0 * x[0] + x[1]);
        auto y2 = log(y1);
        auto y = (y1 + x[3] * y2) * (y1 + y2);
        return y;
    }

    template <class T>
    T fStarter(const std::vector<T> &a)
    {
        auto b1 = a[0] + exp(a[1]);
        auto b2 = sin(a[2]) + cos(a[3]);
        auto b3 = pow(a[1], 1.5) + a[3];
        auto b4 = cos(b1) * b2 + b3;
        return b4;
    }

    template <class T>
    T bs_price(T forward, T volatility, T numeraire, T strike, T expiry, bool isCall)
    {
        auto periodVolatility = volatility * sqrt(expiry);
        auto dPlus = log(forward / strike) / periodVolatility + 0.5 * periodVolatility;
        auto dMinus = dPlus - periodVolatility;
        auto omega = isCall ? 1.0 : -1.0;
        auto nPlus = normalCdf(omega * dPlus);
        auto nMinus = normalCdf(omega * dMinus);
        auto price = numeraire * omega * (forward * nPlus - strike * nMinus);
        return price;
    }

    Number _Z_RANGE = 1.0E-6;

    template <typename T>
    T sabr_volatility(T forward, T alpha, T beta,
                      T rho, T nu, T strike, T expiry)
    {
        T beta1 = 1.0 - beta;
        T fKbeta = pow(forward * strike, 0.5 * beta1);
        T logfK = log(forward / strike);
        T z = nu / alpha * fKbeta * logfK;
        T zxz = 0.0;
        T xz = 0.0;
        T sqz = 0.0;
        if (fabs(z) < _Z_RANGE)
        {
            zxz = 1.0 - 0.5 * z * rho;
        }
        else
        {
            sqz = sqrt(1.0 - 2.0 * rho * z + z * z);
            xz = log((sqz + z - rho) / (1.0 - rho));
            zxz = z / xz;
        }
        T beta24 = beta1 * beta1 / 24.0;
        T beta1920 = beta1 * beta1 * beta1 * beta1 / 1920.0;
        T logfK2 = logfK * logfK;
        T factor11 = beta24 * logfK2;
        T factor12 = beta1920 * logfK2 * logfK2;
        T num1 = (1 + factor11 + factor12);
        T factor1 = alpha / (fKbeta * num1);
        T factor31 = beta24 * alpha * alpha / (fKbeta * fKbeta);
        T factor32 = 0.25 * rho * beta * nu * alpha / fKbeta;
        T factor33 = (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu;
        T factor3 = 1 + (factor31 + factor32 + factor33) * expiry;
        return factor1 * zxz * factor3;
    }

    template <typename T>
    T sabr_price(T forward, T alpha, T beta, T rho, T nu, T numeraire, 
        T strike, T expiry, bool isCall)
    {
        auto volatility = sabr_volatility(forward, alpha, beta, rho, nu, strike, expiry);            
        auto price = bs_price(forward, volatility, numeraire, strike, expiry, isCall);
        return price;
    }

}

TEST(BasicInstrumentation, BasicInstrumentationTest01)
{
    std::println("BasicInstrumentationTest");
    Number::tape->rewind();

    Number x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    x->putOnTape();

    // Evaluate and build the tape
    Number y = f(x);

    // Propagate adjoints through the tape in reverse order
    y.propagateToStart();

    // Get derivatives
    // 950.736, 190.147, 443.677, 73.2041, 0
    for (size_t i = 0; i < 5; ++i)
    {
        std::println("a: {}, = {}", i, x[i].adjoint());
    }
}

TEST(BasicInstrumentation, BasicInstrumentationTest02)
{

    std::println("BasicInstrumentationTest02");
    Number::tape->rewind();

    std::vector<Number> x{1.0, 2.0, 3.0, 4.0};
    for (auto &num : x)
    {
        num.putOnTape();
    }

    // Evaluate and build the tape
    Number y = fStarter(x);

    // Propagate adjoints through the tape in reverse order
    y.propagateToStart();

    // Get derivatives
    // 0.440889 5.37907 0.504802 0.614103
    for (size_t i = 0; i < x.size(); ++i)
    {
        std::println("a: {}, = {}", i, x[i].adjoint());
    }
}

TEST(BasicInstrumentation, BlackFormualTest)
{
    std::println("BlackFormualTest");
    Number::tape->rewind();

    Number forward = 100.0;
    Number volatility = 0.2;
    Number numeraire = 1.0;
    Number strike = 100.0;
    Number expiry = 1.0;
    bool isCall = true;

    forward.putOnTape();
    volatility.putOnTape();
    numeraire.putOnTape();
    strike.putOnTape();
    expiry.putOnTape();

    // Evaluate and build the tape
    Number y = bs_price(forward, volatility, numeraire, strike, expiry, isCall);

    // Propagate adjoints through the tape in reverse order
    y.propagateToStart();

    // Get derivatives
    // Derivative: 0.539828 39.6953 7.96557 -0.460172 3.96953
    std::println("forward: = {}", forward.adjoint());
    std::println("volatility: = {}", volatility.adjoint());
    std::println("numeraire: = {}", numeraire.adjoint());
    std::println("strike: = {}", strike.adjoint());
    std::println("expiry: = {}", expiry.adjoint());
}

TEST(BasicInstrumentation, SabrVolatilityFormulaTest)
{
    std::println("SabrVolatilityFormulaTest");
    Number::tape->rewind();

    Number forward = 0.03;
    Number alpha = 0.2;
    Number beta = 0.5;
    Number rho = -0.3;
    Number nu = 0.4;
    Number strike = 0.03;
    Number expiry = 1.0;

    forward.putOnTape();
    alpha.putOnTape();
    beta.putOnTape();
    rho.putOnTape();
    nu.putOnTape();
    strike.putOnTape();
    expiry.putOnTape();

    // Evaluate and build the tape
    Number vol = sabr_volatility(forward, alpha, beta, rho, nu, strike, expiry);
    // vol = 1.16406
    std::println("vol: = {}", vol.value());

    // Propagate adjoints through the tape in reverse order
    vol.propagateToStart();

    // Get derivatives
    // Derivative: 1.16406
    std::println("vol_adjoint: = {}", vol.adjoint());
}

TEST(BasicInstrumentation, SabrPriceFormulaTest)
{
    std::println("==========================");
    std::println("SabrPriceFormulaTest");
    std::println("==========================");

    Number::tape->rewind();

    Number forward = 100.0;
    Number alpha = 0.2;
    Number beta = 0.5;
    Number rho = -0.3;
    Number nu = 0.4;
    Number numeraire = 1.0;
    Number strike = 100.0;
    Number expiry = 1.0;
    bool isCall = true;

    forward.putOnTape();
    alpha.putOnTape();
    beta.putOnTape();
    rho.putOnTape();
    nu.putOnTape();
    strike.putOnTape();
    expiry.putOnTape();

    // Evaluate and build the tape
    Number y = sabr_price(forward, alpha, beta, rho, nu,
        numeraire, strike, expiry, isCall);

    // Propagate adjoints through the tape in reverse order
    y.propagateToStart();

    // Get derivatives
    // Derivative: 0.477813 4.03288 3.71393 0.010372 0.0454106 0.806837 -0.473778 0.412371
    std::println("forward: = {}", forward.adjoint());
    std::println("alpha: = {}", alpha.adjoint());
    std::println("beta: = {}", beta.adjoint());
    std::println("rho: = {}", rho.adjoint());
    std::println("nu: = {}", nu.adjoint());
    std::println("numeraire: = {}", numeraire.adjoint());
    std::println("strike: = {}", strike.adjoint());
    std::println("expiry: = {}", expiry.adjoint());
    std::println("");
}
