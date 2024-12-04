#if !defined(AD_STARTER_H_ALREADY_INCLUDED)
#define AD_STARTER_H_ALREADY_INCLUDED

#include <iostream>
#include <cmath>
#include <vector>
#include "Utils.h"

class AdStarter
{
public:
    static double f(const std::vector<double> &a);
    DoubleDerivatives f_Sad(const std::vector<double> &a);
    static DoubleDerivatives f_Aad(const std::vector<double> &a);
};

#endif // AD_STARTER_H_ALREADY_INCLUDED