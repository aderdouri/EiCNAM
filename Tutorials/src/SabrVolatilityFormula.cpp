#include "Utils.h"
#include "SabrVolatilityFormula.h"

double SabrVolatilityFormula::volatility(double forward, double alpha, double beta,
                                         double rho, double nu, double strike, double expiry)
{
    double beta1 = 1.0 - beta;
    double fKbeta = std::pow(forward * strike, 0.5 * beta1);
    double logfK = std::log(forward / strike);
    double z = nu / alpha * fKbeta * logfK;
    double zxz;
    double xz = 0.0;
    double sqz = 0.0;
    if (std::abs(z) < Z_RANGE)
    {
        zxz = 1.0 - 0.5 * z * rho;
    }
    else
    {
        sqz = std::sqrt(1.0 - 2.0 * rho * z + z * z);
        xz = std::log((sqz + z - rho) / (1.0 - rho));
        zxz = z / xz;
    }
    double beta24 = beta1 * beta1 / 24.0;
    double beta1920 = beta1 * beta1 * beta1 * beta1 / 1920.0;
    double logfK2 = logfK * logfK;
    double factor11 = beta24 * logfK2;
    double factor12 = beta1920 * logfK2 * logfK2;
    double num1 = (1 + factor11 + factor12);
    double factor1 = alpha / (fKbeta * num1);
    double factor31 = beta24 * alpha * alpha / (fKbeta * fKbeta);
    double factor32 = 0.25 * rho * beta * nu * alpha / fKbeta;
    double factor33 = (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu;
    double factor3 = 1 + (factor31 + factor32 + factor33) * expiry;
    return factor1 * zxz * factor3;
}

DoubleDerivatives SabrVolatilityFormula::volatility_Aad(double forward, double alpha, double beta, double rho, double nu, double strike, double expiry)
{
    double beta1 = 1.0 - beta;
    double fKbeta = std::pow(forward * strike, 0.5 * beta1);
    double logfK = std::log(forward / strike);
    double z = nu / alpha * fKbeta * logfK;
    double zxz;
    double xz = 0.0;
    double sqz = 0.0;
    if (std::abs(z) < Z_RANGE)
    {
        zxz = 1.0 - 0.5 * z * rho;
    }
    else
    {
        sqz = std::sqrt(1.0 - 2.0 * rho * z + z * z);
        xz = std::log((sqz + z - rho) / (1.0 - rho));
        zxz = z / xz;
    }
    double beta24 = beta1 * beta1 / 24.0;
    double beta1920 = beta1 * beta1 * beta1 * beta1 / 1920.0;
    double logfK2 = logfK * logfK;
    double factor11 = beta24 * logfK2;
    double factor12 = beta1920 * logfK2 * logfK2;
    double num1 = (1 + factor11 + factor12);
    double factor1 = alpha / (fKbeta * num1);
    double factor31 = beta24 * alpha * alpha / (fKbeta * fKbeta);
    double factor32 = 0.25 * rho * beta * nu * alpha / fKbeta;
    double factor33 = (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu;
    double factor3 = 1 + (factor31 + factor32 + factor33) * expiry;
    double volatility = factor1 * zxz * factor3;

    double volatilityBar = 1.0;
    double factor3Bar = factor1 * zxz * volatilityBar;
    double factor33Bar = expiry * factor3Bar;
    double factor32Bar = expiry * factor3Bar;
    double factor31Bar = expiry * factor3Bar;
    double factor1Bar = zxz * factor3 * volatilityBar;
    double num1Bar = -alpha / (fKbeta * num1 * num1) * factor1Bar;
    double factor12Bar = num1Bar;
    double factor11Bar = num1Bar;
    double logfK2Bar = beta24 * factor11Bar;
    logfK2Bar += 2.0 * beta1920 * logfK2 * factor12Bar;
    double beta1920Bar = logfK2 * logfK2 * factor12Bar;
    double beta24Bar = logfK2 * factor11Bar;
    beta24Bar += alpha * alpha / (fKbeta * fKbeta) * factor31Bar;
    double zxzBar = factor1 * factor3 * volatilityBar;
    double zBar;
    double xzBar = 0.0;
    double sqzBar = 0.0;
    if (std::abs(z) < Z_RANGE)
    {
        zBar = 0.5 * rho * zxzBar;
    }
    else
    {
        xzBar = -z / (xz * xz) * zxzBar;
        sqzBar = xzBar / (sqz + z - rho);
        zBar = zxzBar / xz;
        zBar += xzBar / (sqz + z - rho);
        zBar += (-rho + z) / sqz * sqzBar;
    }
    double logfKBar = nu / alpha * fKbeta * zBar;
    logfKBar += 2.0 * logfK * logfK2Bar;
    double fKbetaBar = nu / alpha * logfK * zBar;
    fKbetaBar += -alpha / (fKbeta * fKbeta * num1) * factor1Bar;
    fKbetaBar += -2.0 * beta24 * alpha * alpha / (fKbeta * fKbeta * fKbeta) * factor31Bar;
    fKbetaBar += -0.25 * rho * beta * nu * alpha / (fKbeta * fKbeta) * factor32Bar;
    double beta1Bar = fKbeta * 0.5 * std::log(forward * strike) * fKbetaBar;
    beta1Bar += beta1 / 12.0 * beta24Bar;
    beta1Bar += beta1 * beta1 * beta1 / 480.0 * beta1920Bar;
    std::vector<double> inputBar(7, 0.0);
    inputBar[0] += logfKBar / forward;
    inputBar[0] += 0.5 * beta1 * fKbeta / forward * fKbetaBar;
    inputBar[1] += -nu / (alpha * alpha) * fKbeta * logfK * zBar;
    inputBar[1] += factor1Bar / (fKbeta * num1);
    inputBar[1] += 2.0 * beta24 * alpha / (fKbeta * fKbeta) * factor31Bar;
    inputBar[1] += 0.25 * rho * beta * nu / fKbeta * factor32Bar;
    inputBar[2] += -beta1Bar;
    inputBar[2] += 0.25 * rho * nu * alpha / fKbeta * factor32Bar;
    if (std::abs(z) < Z_RANGE)
    {
        inputBar[3] += -0.5 * z * zxzBar;
    }
    else
    {
        inputBar[3] += -z / sqz * sqzBar;
        inputBar[3] += (-1.0 / (sqz + z - rho) + 1.0 / (1.0 - rho)) * xzBar;
    }
    inputBar[3] += 0.25 * beta * nu * alpha / fKbeta * factor32Bar;
    inputBar[3] += -0.25 * rho * nu * nu * factor33Bar;
    inputBar[4] += fKbeta / alpha * logfK * zBar;
    inputBar[4] += 0.25 * rho * beta * alpha / fKbeta * factor32Bar;
    inputBar[4] += (2.0 - 3.0 * rho * rho) / 12.0 * nu * factor33Bar;
    inputBar[5] += 0.5 * beta1 * fKbeta / strike * fKbetaBar;
    inputBar[5] += -logfKBar / strike;
    inputBar[6] += (factor31 + factor32 + factor33) * factor3Bar;
    return DoubleDerivatives(volatility, inputBar);
}
