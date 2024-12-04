int main()
{
    // Factorial
    constexpr auto factorial = [](this const auto self, const auto n)
    {
        if (n < 2)
        {
            return 1;
        }
        else
        {
            return n * self(n - 1); // same as: return n * factorial(n - 1);
        }
    };

    static_assert(factorial(5) == 120, "factorial() incorrect");

    return 0;
}
