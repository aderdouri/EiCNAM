import torch

def longstaff_schwartz_best_of_two(S0_1, S0_2, K, sigma1, sigma2, rho, T, r, Np, NT):
    """
    Longstaff-Schwartz algorithm for the best of two assets implemented in PyTorch.

    Args:
        S0_1: Initial price of asset 1.
        S0_2: Initial price of asset 2.
        K: Strike price.
        sigma1: Volatility of asset 1.
        sigma2: Volatility of asset 2.
        rho: Correlation between the two assets.
        T: Time to maturity.
        r: Risk-free rate.
        Np: Number of simulated paths.
        NT: Number of time steps.

    Returns:
        V: Option value.
    """
    dt = T / torch.tensor(NT, dtype=torch.float32)  # Ensure dt is a tensor
    sqrt_dt = torch.sqrt(dt)
    #rho = torch.tensor(rho, dtype=torch.float32)  # Ensure rho is a tensor

    # Simulate paths
    Z1 = torch.randn(Np, NT)
    Z2 = rho * Z1 + torch.sqrt(torch.tensor(1.0, dtype=torch.float32) - rho**2) * torch.randn(Np, NT)
    Sp1 = torch.zeros(Np, NT, dtype=torch.float32)
    Sp2 = torch.zeros(Np, NT, dtype=torch.float32)
    Sp1[:, 0] = S0_1
    Sp2[:, 0] = S0_2

    for t in range(1, NT):
        previous_step1 = Sp1[:, t - 1].clone()  # Avoid modifying previous step
        previous_step2 = Sp2[:, t - 1].clone()  # Avoid modifying previous step
        Sp1[:, t] = previous_step1 * torch.exp((r - 0.5 * sigma1**2) * dt + sigma1 * sqrt_dt * Z1[:, t])
        Sp2[:, t] = previous_step2 * torch.exp((r - 0.5 * sigma2**2) * dt + sigma2 * sqrt_dt * Z2[:, t])

    # Initialize cash flows
    cash_flow = torch.maximum(torch.maximum(Sp1[:, -1], Sp2[:, -1]) - K, torch.tensor(0.0, dtype=torch.float32))
    discount_factor = torch.exp(-r * dt)

    # Backward induction
    cash_flow = cash_flow.clone()  # Ensure no inplace modification
    for t in range(NT - 2, 0, -1):
        in_the_money = torch.maximum(Sp1[:, t], Sp2[:, t]) > K
        itm_indices = torch.where(in_the_money)[0]

        if len(itm_indices) > 0:
            X1 = Sp1[itm_indices, t]
            X2 = Sp2[itm_indices, t]
            Y = cash_flow[itm_indices] * discount_factor.clone()

            # Regression to approximate continuation value
            A = torch.stack([torch.ones_like(X1), X1, X1**2, X2, X2**2], dim=1)
            coeffs = torch.linalg.lstsq(A, Y).solution

            continuation_value = coeffs[0] + coeffs[1] * X1 + coeffs[2] * X1**2 + coeffs[3] * X2 + coeffs[4] * X2**2

            exercise_value = torch.maximum(X1, X2) - K

            exercise = exercise_value > continuation_value
            exercise_indices = itm_indices[exercise]

            cash_flow = cash_flow.clone()  # Avoid inplace modification
            cash_flow[exercise_indices] = exercise_value[exercise]

        cash_flow = cash_flow * discount_factor.clone()  # Ensure no inplace modification

    # Final option value
    V = cash_flow.mean() * torch.exp(-r * dt)
    return V

def calculate_sensitivities_best_of_two(S0_1, S0_2, K, sigma1, sigma2, rho, T, r, Np, NT):
    """
    Calculate sensitivities (Delta1, Delta2, Vega1, Vega2, Rho, Theta) for the best of two assets using automatic differentiation.

    Args:
        S0_1: Initial price of asset 1.
        S0_2: Initial price of asset 2.
        K: Strike price.
        sigma1: Volatility of asset 1.
        sigma2: Volatility of asset 2.
        rho: Correlation between the two assets.
        T: Time to maturity.
        r: Risk-free rate.
        Np: Number of simulated paths.
        NT: Number of time steps.

    Returns:
        sensitivities: Dictionary containing Delta1, Delta2, Vega1, Vega2, Rho, Theta.
    """
    S0_1_t = torch.tensor(S0_1, requires_grad=True, dtype=torch.float32)
    S0_2_t = torch.tensor(S0_2, requires_grad=True, dtype=torch.float32)
    sigma1_t = torch.tensor(sigma1, requires_grad=True, dtype=torch.float32)
    sigma2_t = torch.tensor(sigma2, requires_grad=True, dtype=torch.float32)
    rho_t = torch.tensor(rho, requires_grad=True, dtype=torch.float32)
    r_t = torch.tensor(r, requires_grad=True, dtype=torch.float32)
    T_t = torch.tensor(T, requires_grad=True, dtype=torch.float32)

    # Enable anomaly detection
    with torch.autograd.set_detect_anomaly(True):
        # Compute option value
        V = longstaff_schwartz_best_of_two(S0_1_t, S0_2_t, K, sigma1_t, sigma2_t, rho_t, T_t, r_t, Np, NT)

        # Compute gradients
        V.backward()

    delta1 = S0_1_t.grad.item()
    delta2 = S0_2_t.grad.item()
    vega1 = sigma1_t.grad.item()
    vega2 = sigma2_t.grad.item()
    rho = rho_t.grad.item()
    theta = T_t.grad.item()

    return {
        "Delta1": delta1,
        "Delta2": delta2,
        "Vega1": vega1,
        "Vega2": vega2,
        "Rho": rho,
        "Theta": theta
    }

# Parameters for testing
S0_1 = 100.0
S0_2 = 105.0
K = 95.0
sigma1 = 0.25
sigma2 = 0.30
rho = 0.5
T = 180 / 365
r = 0.05
Np = 100
NT = 1000

# Test the algorithm
option_value = longstaff_schwartz_best_of_two(S0_1, S0_2, K, sigma1, sigma2, rho, T, r, Np, NT)
sensitivities = calculate_sensitivities_best_of_two(S0_1, S0_2, K, sigma1, sigma2, rho, T, r, Np, NT)

print(f"Option Value: {option_value}")
print(f"Sensitivities: {sensitivities}")