import torch

def simulate_intensity_path(lambda_0, k, mu, nu, h, M, Np):
    """
    Simulates intensity paths using a CIR-like stochastic model.

    Args:
        lambda_0: Initial intensity.
        k: Speed of mean reversion.
        mu: Long-term mean intensity.
        nu: Volatility of intensity.
        h: Time step size.
        M: Number of exercise opportunities.
        Np: Number of paths.

    Returns:
        lambda_paths: Simulated intensity paths (Np x M).
    """
    lambda_paths = torch.zeros(Np, M, dtype=torch.float32)
    lambda_paths[:, 0] = lambda_0

    Z = torch.randn(Np, M - 1, dtype=torch.float32)  # Random normal increments
    for m in range(1, M):
        lambda_prev = lambda_paths[:, m - 1]
        lambda_paths[:, m] = lambda_prev + k * (mu - lambda_prev) * h + nu * torch.sqrt(lambda_prev * h) * Z[:, m - 1]
        lambda_paths[:, m] = torch.maximum(lambda_paths[:, m], torch.tensor(0.0))  # Ensure non-negative intensity

    return lambda_paths


def calculate_survival_probabilities(lambda_paths, time_steps):
    """
    Calculates survival probabilities from intensity paths.

    Args:
        lambda_paths: Simulated intensity paths (Np x M).
        time_steps: Time steps for the exercise points.

    Returns:
        survival_probs: Survival probabilities (Np x M).
    """
    cumulative_intensity = torch.cumsum(lambda_paths * time_steps.diff(), dim=1)
    survival_probs = torch.exp(-cumulative_intensity)
    return survival_probs


def calculate_cva_and_option_price(S0, K, T, r, sigma, M, LGD, lambda_0, k, mu, nu, Np, NT):
    """
    Calculates CVA and option price for a Bermudan option using a stochastic intensity model.

    Args:
        S0: Initial stock price.
        K: Strike price of the option.
        T: Maturity.
        r: Risk-free rate.
        sigma: Volatility of the underlying.
        M: Number of exercise opportunities.
        LGD: Loss Given Default.
        lambda_0: Initial intensity.
        k: Speed of mean reversion.
        mu: Long-term mean intensity.
        nu: Volatility of intensity.
        Np: Number of Monte Carlo paths.
        NT: Number of time steps for the underlying simulation.

    Returns:
        CVA: Credit Valuation Adjustment.
        option_price: Price of the Bermudan option.
    """
    h = T / M  # Time step for intensity simulation
    time_steps = torch.linspace(0, T, M + 1)

    # Simulate intensity paths
    lambda_paths = simulate_intensity_path(lambda_0, k, mu, nu, h, M, Np)

    # Compute survival probabilities
    survival_probs = calculate_survival_probabilities(lambda_paths, time_steps)

    # Simulate underlying asset paths
    dt = T / NT
    sqrt_dt = torch.sqrt(torch.tensor(dt, dtype=torch.float32))
    Z = torch.randn(Np, NT)
    S = torch.zeros(Np, NT + 1, dtype=torch.float32)
    S[:, 0] = S0
    for t in range(1, NT + 1):
        S[:, t] = S[:, t - 1] * torch.exp((r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z[:, t - 1])

    # Bermudan option valuation
    exercise_indices = torch.linspace(0, NT, M + 1).long()[1:]  # Map exercise times to simulation indices
    option_values = torch.zeros(Np, M + 1, dtype=torch.float32)
    payoff = torch.maximum(K - S[:, -1], torch.tensor(0.0))
    option_values[:, -1] = payoff

    h_tensor = torch.tensor(h, dtype=torch.float32)
    for m in range(M - 1, -1, -1):
        t_idx = exercise_indices[m]
        itm = K > S[:, t_idx]
        itm_indices = torch.where(itm)[0]
        if itm_indices.numel() > 0:
            X = S[itm_indices, t_idx]
            Y = option_values[itm_indices, m + 1] * torch.exp(-r * h_tensor)
            A = torch.stack([torch.ones_like(X), X, X**2], dim=1)
            coeffs = torch.linalg.lstsq(A, Y).solution
            continuation_value = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2
            exercise_value = K - X
            exercise = exercise_value > continuation_value
            option_values[itm_indices[exercise], m] = exercise_value[exercise]
            option_values[itm_indices[~exercise], m] = continuation_value[~exercise]

    # Calculate CVA
    CVA = 0.0
    for m in range(1, M):
        discounted_value = option_values[:, m] * torch.exp(-r * time_steps[m])
        default_prob = survival_probs[:, m - 1] - survival_probs[:, m]
        CVA += LGD * (default_prob * discounted_value).mean()

    # Calculate option price
    option_price = torch.exp(-r * time_steps[1:]) * option_values[:, :-1]
    option_price = option_price.mean(dim=0).sum().item()

    return CVA.item(), option_price

# Parameters
# Updated Parameters
S0 = 1.0         # Initial stock price
K = 0.9          # Strike price
T = 1.0          # Maturity (years)
r = 0.15         # Risk-free rate
sigma = 0.2      # Volatility
M = 12           # Number of exercise opportunities (quarterly)
LGD = 0.6        # Loss Given Default
lambda_0 = 1.0   # Initial intensity
k = 0.5          # Speed of mean reversion
mu = 1.0         # Long-term mean intensity
nu = 0.25        # Volatility of intensity
Np = 10000       # Number of Monte Carlo paths
NT = 1000        # Number of time steps for underlying

# Calculate CVA and option price
cva_value, option_price = calculate_cva_and_option_price(S0, K, T, r, sigma, M, LGD, lambda_0, k, mu, nu, Np, NT)
print(f"Option Price: {option_price:.6f}")
print(f"CVA: {cva_value:.6f}")

def finite_difference_sensitivities(S0, K, T, r, sigma, M, LGD, lambda_0, k, mu, nu, Np, NT, delta=1e-4):
    """
    Compute sensitivities of CVA and option price using finite differences.

    Args:
        S0: Initial stock price.
        K: Strike price.
        T: Maturity of the option.
        r: Risk-free rate.
        sigma: Volatility of the underlying.
        M: Number of exercise opportunities.
        LGD: Loss Given Default.
        lambda_0: Initial intensity.
        k: Speed of mean reversion.
        mu: Long-term mean intensity.
        nu: Volatility of intensity.
        Np: Number of Monte Carlo paths.
        NT: Number of time steps for underlying simulation.
        delta: Small change for finite difference approximation.

    Returns:
        sensitivities: Dictionary containing Delta, Vega, Rho, and Lambda_0 sensitivities for both CVA and option price.
    """
    # Compute CVA and option price for original parameters
    CVA_original, option_price_original = calculate_cva_and_option_price(S0, K, T, r, sigma, M, LGD, lambda_0, k, mu, nu, Np, NT)
    
    # Delta (sensitivity to initial stock price)
    CVA_S_up, option_price_S_up = calculate_cva_and_option_price(S0 + delta, K, T, r, sigma, M, LGD, lambda_0, k, mu, nu, Np, NT)
    delta_S_CVA = (CVA_S_up - CVA_original) / delta
    delta_S_option = (option_price_S_up - option_price_original) / delta

    # Vega (sensitivity to volatility)
    CVA_sigma_up, option_price_sigma_up = calculate_cva_and_option_price(S0, K, T, r, sigma + delta, M, LGD, lambda_0, k, mu, nu, Np, NT)
    delta_sigma_CVA = (CVA_sigma_up - CVA_original) / delta
    delta_sigma_option = (option_price_sigma_up - option_price_original) / delta

    # Rho (sensitivity to risk-free rate)
    CVA_r_up, option_price_r_up = calculate_cva_and_option_price(S0, K, T, r + delta, sigma, M, LGD, lambda_0, k, mu, nu, Np, NT)
    delta_r_CVA = (CVA_r_up - CVA_original) / delta
    delta_r_option = (option_price_r_up - option_price_original) / delta

    # Lambda_0 (sensitivity to initial intensity)
    CVA_lambda_up, option_price_lambda_up = calculate_cva_and_option_price(S0, K, T, r, sigma, M, LGD, lambda_0 + delta, k, mu, nu, Np, NT)
    delta_lambda_CVA = (CVA_lambda_up - CVA_original) / delta
    delta_lambda_option = (option_price_lambda_up - option_price_original) / delta

    return {
        "CVA": {
            "Delta": delta_S_CVA,
            "Vega": delta_sigma_CVA,
            "Rho": delta_r_CVA,
            "Lambda_0": delta_lambda_CVA,
        },
        "Option Price": {
            "Delta": delta_S_option,
            "Vega": delta_sigma_option,
            "Rho": delta_r_option,
            "Lambda_0": delta_lambda_option,
        }
    }

# Compute sensitivities using finite differences
sensitivities = finite_difference_sensitivities(S0, K, T, r, sigma, M, LGD, lambda_0, k, mu, nu, Np, NT)

# Print sensitivities
print("Sensitivities for CVA:")
print(f"Delta: {sensitivities['CVA']['Delta']:.6f}")
print(f"Vega: {sensitivities['CVA']['Vega']:.6f}")
print(f"Rho: {sensitivities['CVA']['Rho']:.6f}")
print(f"Lambda_0: {sensitivities['CVA']['Lambda_0']:.6f}")

print("Sensitivities for Option Price:")
print(f"Delta: {sensitivities['Option Price']['Delta']:.6f}")
print(f"Vega: {sensitivities['Option Price']['Vega']:.6f}")
print(f"Rho: {sensitivities['Option Price']['Rho']:.6f}")
print(f"Lambda_0: {sensitivities['Option Price']['Lambda_0']:.6f}")
