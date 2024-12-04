#include <iostream>

// Base class that provides increment operators
struct Incrementable
{
    // Prefix increment: ++self
    auto &operator++(this auto &self)
    {
        self.setValue(self.getValue() + 1);
        return self;
    }

    // Postfix increment: self++
    auto operator++(this auto &self, int)
    {
        auto tmp = self; // Copy the current state
        self.setValue(self.getValue() + 1);
        return tmp; // Return the original state
    }
};

// Derived class: Counter
struct Counter : Incrementable
{
    std::size_t getValue() const { return value; }
    void setValue(std::size_t newValue) { value = newValue; }

private:
    std::size_t value = 0;
};

// Derived class: Age
struct Age : Incrementable
{
    explicit Age(unsigned short initialValue) : value(initialValue) {}

    unsigned short getValue() const { return value; }
    void setValue(unsigned short newValue) { value = newValue; }

private:
    unsigned short value = 0;
};

int main()
{
    Age a(38);
    ++a;                               // Increment Age
    std::cout << a.getValue() << '\n'; // Prints: 39
    return 0;
}
