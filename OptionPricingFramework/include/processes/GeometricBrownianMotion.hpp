/* GeometricBrownianMotion.hpp - GBM Process */
#ifndef GEOMETRIC_BROWNIAN_MOTION_HPP
#define GEOMETRIC_BROWNIAN_MOTION_HPP

#include <random>

class GeometricBrownianMotion
{
private:
    double drift, volatility;

public:
    GeometricBrownianMotion(double mu, double sigma) : drift(mu), volatility(sigma) {}

    template <typename RNG, typename DIST>
    double simulate(double S0, double T, int steps, RNG &rng, DIST &norm_dist) const
    {
        double dt = T / steps;
        double S = S0;
        for (int i = 0; i < steps; ++i)
        {
            double dW = sqrt(dt) * norm_dist(rng);
            S *= exp((drift - 0.5 * volatility * volatility) * dt + volatility * dW);
        }
        return S;
    }
};

#endif // GEOMETRIC_BROWNIAN_MOTION_HPP
