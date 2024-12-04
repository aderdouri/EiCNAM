#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <functional>

class Tape
{
public:
    std::vector<std::function<void()>> operations;

    void add_operation(const std::function<void()> &op)
    {
        operations.push_back(op);
    }

    void backward()
    {
        for (auto it = operations.rbegin(); it != operations.rend(); ++it)
        {
            (*it)();
        }
        operations.clear(); // Free memory after backward pass
    }
};

class Variable
{
public:
    double value;
    double gradient;
    static std::shared_ptr<Tape> tape;

    Variable(double val) : value(val), gradient(0.0) {}

    // Addition
    Variable operator+(Variable &other)
    {
        Variable result(value + other.value);
        tape->add_operation([this, &other, &result]()
                            {
            this->gradient += result.gradient;
            other.gradient += result.gradient; });
        return result;
    }

    // Multiplication
    Variable operator*(Variable &other)
    {
        Variable result(value * other.value);
        tape->add_operation([this, &other, &result]()
                            {
            this->gradient += other.value * result.gradient;
            other.gradient += this->value * result.gradient; });
            std::cout << "*: result.value: " << result.value << std::endl;

        return result;
    }

    // Multiplication with Scalar
    Variable operator*(double scalar)
    {
        Variable result(value * scalar);
        std::cout << "* Scalar: result.gradient: " << result.gradient << std::endl; 
        tape->add_operation([this, &scalar, &result]()
                            { this->gradient += scalar * result.gradient; });
        std::cout << "* Scalar: result.value: " << result.value << std::endl;
        std::cout << "* Scalar: result.gradient: " << result.gradient << std::endl;                            
        return result;
    }

    friend Variable operator*(double scalar, Variable &var)
    {
        return var * scalar;
    }
};

Variable exp(Variable &x)
{
    Variable result(std::exp(x.value));
    Variable::tape->add_operation([&x, &result]()
                                  { x.gradient += result.value * result.gradient; });
    return result;
}

// Initialize static member
std::shared_ptr<Tape> Variable::tape = std::make_shared<Tape>();

int main()
{
    Variable x(5.0); // Initial value
    double scalar = 3.0;

    Variable result = x * scalar; // f(x) = x * 3
    result.gradient = 1.0;        // Seed gradient for the result
    Variable::tape->backward();

    std::cout << "Value: " << result.value << std::endl;  // Should be 15.0
    std::cout << "Gradient: " << x.gradient << std::endl; // Should be 3.0

    return 0;
}
