import torch

def european_option_price(S0, K, sigma, T, r, Np, NT):
    """
    Calculate the price of a European call option using Monte Carlo simulation.

    Args:
        S0: Initial asset price.
        K: Strike price.
        sigma: Volatility.
        T: Time to maturity.
        r: Risk-free rate.
        Np: Number of simulated paths.
        NT: Number of time steps.

    Returns:
        option_price: Price of the European call option.
    """
    # Convert constants to PyTorch tensors
    S0 = torch.tensor(S0, dtype=torch.float64)
    r = torch.tensor(r, dtype=torch.float64)
    T = torch.tensor(T, dtype=torch.float64)
    sigma = torch.tensor(sigma, dtype=torch.float64)
    K = torch.tensor(K, dtype=torch.float64)

    # Time step size
    dt = T / NT
    sqrt_dt = torch.sqrt(dt)

    # Simulate paths
    Z = torch.randn(Np, NT, dtype=torch.float64)
    Sp = torch.zeros(Np, NT, dtype=torch.float64)
    Sp[:, 0] = S0

    for t in range(1, NT):
        Sp[:, t] = Sp[:, t - 1] * torch.exp((r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z[:, t])

    # Calculate payoff for European call option
    payoff = torch.maximum(Sp[:, -1] - K, torch.tensor(0.0, dtype=torch.float64))

    # Discount payoff back to present value
    discount_factor = torch.exp(-r * T)
    discounted_payoff = payoff * discount_factor

    # Compute option price
    option_price = discounted_payoff.mean()

    return option_price.item()

# Parameters for testing
S0 = 100.0
K = 90.0
sigma = 0.25
T = 2.0
r = 0.01
Np = 500000  # Number of paths
NT = 1000    # Number of time steps

# Test the function
option_price = european_option_price(S0, K, sigma, T, r, Np, NT)
print(f"European Call Option Price: {option_price:.4f}")


# Given constants
k = 0.5
mu = 1.0
nu = 0.25
lambda0 = 1.0
LGD = 0.6

# Convert constants to PyTorch tensors
k = torch.tensor(k, dtype=torch.float64)
mu = torch.tensor(mu, dtype=torch.float64)
nu = torch.tensor(nu, dtype=torch.float64)

# Compute h
h = torch.sqrt(k**2 + 2 * nu**2)

# Compute A(0, T)
numerator_A = 2 * h * torch.exp((k + h) * T / 2)
denominator_A = 2 * h + (k + h) * (torch.exp(h * T) - 1)
exponent_A = (2 * k * mu) / (nu**2)
A_0_T = (numerator_A / denominator_A)**exponent_A

# Compute B(0, T)
numerator_B = 2 * h * (torch.exp(h * T) - 1)
denominator_B = 2 * h + (k + h) * (torch.exp(h * T) - 1)
B_0_T = numerator_B / denominator_B

# Print results
print(f"h: {h.item()}")
print(f"A(0, T): {A_0_T.item()}")
print(f"B(0, T): {B_0_T.item()}")

CVA = LGD * (1 - A_0_T * torch.exp(-lambda0 * B_0_T)) * option_price    

print(f"CVA: {CVA.item()}")


# Compute CVA sensitivity with respect to λ0
p_bar = 1.0
u_bar = LGD * option_price * p_bar 
lambda0_bar = A_0_T * B_0_T * torch.exp(-B_0_T * lambda0) * u_bar

print(f"CVA Sensitivity with respect to λ0: {lambda0_bar.item()}")