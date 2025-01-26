import torch

def longstaff_schwartz(S0, K, sigma, T, r, Np, NT):
    """
    Longstaff-Schwartz algorithm implemented in PyTorch.

    Args:
        S0: Initial asset price.
        K: Strike price.
        sigma: Volatility.
        T: Time to maturity.
        r: Risk-free rate.
        Np: Number of simulated paths.
        NT: Number of time steps.

    Returns:
        V: Option value.
    """
    dt = T / torch.tensor(NT, dtype=torch.float32)  # Ensure dt is a tensor
    sqrt_dt = torch.sqrt(dt)

    # Simulate paths
    Z = torch.randn(Np, NT)
    Sp = torch.zeros(Np, NT, dtype=torch.float32)
    Sp[:, 0] = S0

    for t in range(1, NT):
        previous_step = Sp[:, t - 1].clone()  # Avoid modifying previous step
        Sp[:, t] = previous_step * torch.exp((r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z[:, t])

    # Initialize cash flows
    cash_flow = torch.maximum(K - Sp[:, -1], torch.tensor(0.0, dtype=torch.float32))
    discount_factor = torch.exp(-r * dt)

    # Backward induction
    cash_flow = cash_flow.clone()  # Ensure no inplace modification
    for t in range(NT - 2, 0, -1):
        in_the_money = Sp[:, t] < K
        itm_indices = torch.where(in_the_money)[0]

        if len(itm_indices) > 0:
            X = Sp[itm_indices, t]
            Y = cash_flow[itm_indices] * discount_factor.clone()

            # Regression to approximate continuation value
            A = torch.stack([torch.ones_like(X), X, X**2], dim=1)
            coeffs = torch.linalg.lstsq(A, Y).solution

            continuation_value = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2

            exercise_value = K - X

            exercise = exercise_value > continuation_value
            exercise_indices = itm_indices[exercise]

            cash_flow = cash_flow.clone()  # Avoid inplace modification
            cash_flow[exercise_indices] = exercise_value[exercise]

        cash_flow = cash_flow * discount_factor.clone()  # Ensure no inplace modification

    # Final option value
    V = cash_flow.mean() * torch.exp(-r * dt)
    return V

def calculate_sensitivities(S0, K, sigma, T, r, Np, NT):
    """
    Calculate sensitivities (Delta, Vega, Rho, Theta) using automatic differentiation.

    Args:
        S0: Initial asset price.
        K: Strike price.
        sigma: Volatility.
        T: Time to maturity.
        r: Risk-free rate.
        Np: Number of simulated paths.
        NT: Number of time steps.

    Returns:
        sensitivities: Dictionary containing Delta, Vega, Rho, Theta.
    """
    S0_t = torch.tensor(S0, requires_grad=True, dtype=torch.float32)
    sigma_t = torch.tensor(sigma, requires_grad=True, dtype=torch.float32)
    r_t = torch.tensor(r, requires_grad=True, dtype=torch.float32)
    T_t = torch.tensor(T, requires_grad=True, dtype=torch.float32)

    # Enable anomaly detection
    with torch.autograd.set_detect_anomaly(True):
        # Compute option value
        V = longstaff_schwartz(S0_t, K, sigma_t, T_t, r_t, Np, NT)

        # Compute gradients
        V.backward()

    delta = S0_t.grad.item()
    vega = sigma_t.grad.item()
    rho = r_t.grad.item()
    theta = T_t.grad.item()

    return {
        "Delta": delta,
        "Vega": vega,
        "Rho": rho,
        "Theta": theta
    }

# Parameters for testing
S0 = 100.0
K = 95.0
sigma = 0.25
T = 180 / 365
r = 0.05
Np = 50
NT = 1000

# Test the algorithm
option_value = longstaff_schwartz(S0, K, sigma, T, r, Np, NT)
sensitivities = calculate_sensitivities(S0, K, sigma, T, r, Np, NT)

print(f"Option Value: {option_value}")
print(f"Sensitivities: {sensitivities}")



