#include <iostream>
#include <cmath>
#include <concepts>

// Base class for expressions
template <class E>
class Expression
{
};

// Times
// CRTP (Curiously Recurring Template Pattern)
template <typename LHS, typename RHS>
    requires std::derived_from<LHS, Expression<LHS>> && std::derived_from<RHS, Expression<RHS>>
class ExprTimes : public Expression<ExprTimes<LHS, RHS>>
{
    LHS lhs;
    RHS rhs;

public:
    // Constructor
    constexpr explicit ExprTimes(const LHS &l, const RHS &r)
        : lhs(l), rhs(r) {}

    // Function to compute the value
    constexpr double value() const
    {
        return lhs.value() * rhs.value();
    }

    static constexpr int numNumbers = LHS::numNumbers + RHS::numNumbers;
};

// Operator overload for expressions
template <typename LHS, typename RHS>
    requires std::derived_from<LHS, Expression<LHS>> && std::derived_from<RHS, Expression<RHS>>
constexpr auto operator*(const LHS &lhs, const RHS &rhs)
{
    return ExprTimes<LHS, RHS>(lhs, rhs);
}

// Logarithm
template <typename ARG>
    requires std::derived_from<ARG, Expression<ARG>>
class ExprLog : public Expression<ExprLog<ARG>>
{
    ARG arg;

public:
    constexpr explicit ExprLog(const ARG &a) : arg(a) {}

    constexpr double value() const
    {
        return std::log(arg.value());
    }

    static constexpr int numNumbers = ARG::numNumbers;
};

// Operator overload for expressions
template <typename ARG>
    requires std::derived_from<ARG, Expression<ARG>>
constexpr auto log(const ARG &arg)
{
    return ExprLog<ARG>(arg);
}

// Number wrapper
class Number : public Expression<Number>
{
    double val;

public:
    constexpr explicit Number(double v = 0.0) : val(v) {}

    constexpr double value() const
    {
        return val;
    }
    static constexpr int numNumbers = 1;
};

// Generic calculate function
template <class T1, class T2>
constexpr auto calculate(const T1 &t1, const T2 &t2)
{
    return t1 * log(t2);
}

// constexpr function evaluates at compile time
template <class E> 
constexpr auto countNumbersIn (const Expression<E>&)
{
    return E::numNumbers;
}

int main()
{
    Number x1(2.0), x2(3.0);
    auto e = calculate(x1, x2);
    std::cout << e.value() << '\n';
    std::cout << countNumbersIn(e) << '\n'; ;
    return 0;
}