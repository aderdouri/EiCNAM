import torch

def monte_carlo_basket_option(initial_prices, K, sigmas, weights, T, r, Np, NT):
    """
    Monte Carlo simulation for pricing a basket option on multiple independent assets.

    Args:
        initial_prices: Tensor of initial prices of the assets.
        K: Strike price.
        sigmas: Tensor of volatilities of the assets.
        weights: Tensor of weights of the assets in the basket.
        T: Time to maturity.
        r: Risk-free rate.
        Np: Number of simulated paths.
        NT: Number of time steps.

    Returns:
        V: Option value.
    """
    dt = T / NT
    sqrt_dt = torch.sqrt(dt)
    num_assets = initial_prices.shape[0]

    # Simulate paths
    Z = torch.randn(Np, NT, num_assets)
    Sp = torch.zeros(Np, NT, num_assets, dtype=torch.float32)
    Sp[:, 0, :] = initial_prices

    for t in range(1, NT):
        previous_step = Sp[:, t - 1].clone()  # Avoid modifying previous step
        Sp[:, t] = previous_step * torch.exp((r - 0.5 * sigmas**2) * dt + sigmas * sqrt_dt * Z[:, t])

    # Calculate basket value at maturity
    basket_value = (Sp[:, -1, :] * weights).sum(dim=1)
    payoff = torch.maximum(basket_value - K, torch.tensor(0.0, dtype=torch.float32))

    # Discount payoff to present value
    r_tensor = r.clone().detach()
    T_tensor = T.clone().detach()
    V = torch.exp(-r_tensor * T_tensor) * payoff.mean()
    return V

def calculate_sensitivities_basket_option(initial_prices, K, sigmas, weights, T, r, Np, NT):
    """
    Calculate sensitivities (Delta, Vega, Rho, Theta) for the basket option using automatic differentiation.

    Args:
        initial_prices: Tensor of initial prices of the assets.
        K: Strike price.
        sigmas: Tensor of volatilities of the assets.
        weights: Tensor of weights of the assets in the basket.
        T: Time to maturity.
        r: Risk-free rate.
        Np: Number of simulated paths.
        NT: Number of time steps.

    Returns:
        sensitivities: Dictionary containing Delta, Vega, Rho, Theta.
    """
    initial_prices_t = initial_prices.clone().detach().requires_grad_(True)
    sigmas_t = sigmas.clone().detach().requires_grad_(True)
    r_t = r.clone().detach().requires_grad_(True)
    T_t = T.clone().detach().requires_grad_(True)

    # Compute option value
    V = monte_carlo_basket_option(initial_prices_t, K, sigmas_t, weights, T_t, r_t, Np, NT)

    # Compute gradients
    V.backward()

    delta = initial_prices_t.grad  # Gradients w.r.t. initial prices
    vega = sigmas_t.grad  # Gradients w.r.t. volatilities
    rho = r_t.grad.item()  # Gradient w.r.t. risk-free rate
    theta = T_t.grad.item()  # Gradient w.r.t. time to maturity

    return {
        "Delta": delta,
        "Vega": vega,
        "Rho": rho,
        "Theta": theta
    }

# Parameters for testing basket option
initial_prices = torch.tensor([100.0, 82.0, 97.0], dtype=torch.float32)
K = 88.0
sigmas = torch.tensor([0.25, 0.3, 0.1], dtype=torch.float32)
weights = torch.tensor([3.0, 1.0, 2.0], dtype=torch.float32)
T = torch.tensor(2.0, dtype=torch.float32)
r = torch.tensor(0.01, dtype=torch.float32)
Np = 1000
NT = 50000

# Test the basket option pricing
basket_option_value = monte_carlo_basket_option(initial_prices, K, sigmas, weights, T, r, Np, NT)
print(f"Basket Option Value: {basket_option_value.item()}")

# Test the sensitivities
with torch.autograd.set_detect_anomaly(True):
    sensitivities = calculate_sensitivities_basket_option(initial_prices, K, sigmas, weights, T, r, Np, NT)

print("Sensitivities:")
print(f"Delta: {sensitivities['Delta']}")
print(f"Vega: {sensitivities['Vega']}")
print(f"Rho: {sensitivities['Rho']}")
print(f"Theta: {sensitivities['Theta']}")
