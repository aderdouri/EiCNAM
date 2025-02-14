import torch
import numpy as np

def generate_exercise_dates(start_date, maturity_years, frequency, unit):
    if unit == 'months':
        num_exercise_dates = int(maturity_years * 12 / frequency)
        return [start_date + i * frequency / 12 for i in range(1, num_exercise_dates + 1)]
    elif unit == 'weeks':
        num_exercise_dates = int(maturity_years * 52 / frequency)
        return [start_date + i * frequency / 52 for i in range(1, num_exercise_dates + 1)]
    elif unit == 'days':
        num_exercise_dates = int(maturity_years * 365 / frequency)
        return [start_date + i * frequency / 365 for i in range(1, num_exercise_dates + 1)]
    else:
        raise ValueError("unit must be 'months', 'weeks', or 'days'")

def simulate_asset_paths(S0, r, sigma, T, steps, num_paths):
    dt = torch.tensor(T / steps)
    S = torch.zeros(steps + 1, num_paths)
    S[0] = S0
    for t in range(1, steps + 1):
        Z = torch.randn(num_paths)
        S[t] = S[t-1] * torch.exp((r - 0.5 * sigma**2) * dt + sigma * torch.sqrt(dt) * Z)
    return S

def price_bermudan_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='put', steps=500, frequency=3, unit='months', num_paths=500000):
    """
    Pricing a Bermudan Best-of-Two Assets Option using a Monte Carlo method with PyTorch.
    """
    # 1️⃣ Generate exercise dates
    exercise_dates = generate_exercise_dates(0, T, frequency, unit)
    exercise_steps = [int(date * steps / T) for date in exercise_dates]

    # 2️⃣ Simulate asset paths
    S1_paths = simulate_asset_paths(S1, r, sigma1, T, steps, num_paths)
    S2_paths = simulate_asset_paths(S2, r, sigma2, T, steps, num_paths)

    # 3️⃣ Calculate the effective asset paths
    if option_type == 'put':
        Seff_paths = torch.min(S1_paths, S2_paths)
    elif option_type == 'call':
        Seff_paths = torch.max(S1_paths, S2_paths)
    else:
        raise ValueError("option_type must be 'put' or 'call'")

    # 4️⃣ Initialize option values at maturity
    if option_type == 'put':
        option_values = torch.maximum(K - Seff_paths[-1], torch.tensor(0.0))
    elif option_type == 'call':
        option_values = torch.maximum(Seff_paths[-1] - K, torch.tensor(0.0))

    # 5️⃣ Backward induction to price the option
    dt = torch.tensor(T / steps)
    discount_factor = torch.exp(-r * dt)
    for t in reversed(exercise_steps[:-1]):
        continuation_values = discount_factor * option_values
        if option_type == 'put':
            exercise_values = torch.maximum(K - Seff_paths[t], torch.tensor(0.0))
        elif option_type == 'call':
            exercise_values = torch.maximum(Seff_paths[t] - K, torch.tensor(0.0))
        option_values = torch.maximum(exercise_values, continuation_values)

    # 6️⃣ Discount the option values to present value
    option_price = discount_factor * torch.mean(option_values)

    return option_price.item()

# 🔹 Example Pricing for a Bermudan Best-of-Two Assets Option

S1 = 90.0    # Spot price of asset 1
S2 = 100.0   # Spot price of asset 2
K = 100.0    # Strike price
r = 0.04     # Risk-free rate
sigma1 = 0.40 # Volatility of asset 1
sigma2 = 0.40 # Volatility of asset 2
rho = 0.0    # Correlation between assets
T = 1.0      # Time to expiration (1 year)
steps = 500  # Time steps
frequency = 1  # Exercise frequency
unit = 'months'  # Exercise frequency unit
num_paths = 10000  # Number of Monte Carlo paths

# Compute the option price
price = price_bermudan_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='put', steps=steps, 
                                                 frequency=frequency, unit=unit, num_paths=num_paths)
print(f"Bermudan Best-of-Two Assets Put Option Price: {price:.6f}")

price = price_bermudan_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='call', steps=steps, 
                                                 frequency=frequency, unit=unit, num_paths=num_paths)
print(f"Bermudan Best-of-Two Assets Call Option Price: {price:.6f}")


S1 = 1.0    # Spot price of asset 1
S2 = 1.0    # Spot price of asset 2
K = 0.9     # Strike price
r = 0.15      # Risk-free rate
sigma1 = 0.20  # Volatility of asset 1
sigma2 = 0.20 # Volatility of asset 2
rho = 0.0     # Correlation between assets
T = 30.0       # Time to expiration (3 years)
steps = 500   # Binomial tree steps
frequency = 3  # Exercise frequency
unit = 'months'  # Exercise frequency unit

# Compute the option price
price = price_bermudan_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='put', steps=steps, 
                                                 frequency=frequency, unit=unit, num_paths=num_paths)
print(f"Bermudan Best-of-Two Assets Put Option Price: {price:.6f}")

price = price_bermudan_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='call', steps=steps, 
                                                 frequency=frequency, unit=unit, num_paths=num_paths)
print(f"Bermudan Best-of-Two Assets Call Option Price: {price:.6f}")
