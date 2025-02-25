/* EuropeanOption.hpp - European Option Definition */
#ifndef EUROPEAN_OPTION_HPP
#define EUROPEAN_OPTION_HPP

#include <algorithm>

enum OptionType
{
    Call,
    Put
};

class EuropeanOption
{
public:
    double strike, maturity;
    OptionType type;

    EuropeanOption(double K, double T, OptionType optionType) : strike(K), maturity(T), type(optionType) {}

    double payoff(double S) const
    {
        return std::max((type == Call ? S - strike : strike - S), 0.0);
    }
};

#endif // EUROPEAN_OPTION_HPP
