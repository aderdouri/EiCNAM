import torch

def monte_carlo_basket_option(initial_prices, K, sigmas, rho_matrix, weights, T, r, Np, NT):
    """
    Monte Carlo simulation for pricing a basket option on multiple assets.

    Args:
        initial_prices: Tensor of initial prices of the assets.
        K: Strike price.
        sigmas: Tensor of volatilities of the assets.
        rho_matrix: Correlation matrix between the assets.
        weights: Tensor of weights of the assets in the basket.
        T: Time to maturity.
        r: Risk-free rate.
        Np: Number of simulated paths.
        NT: Number of time steps.

    Returns:
        V: Option value.
    """
    dt = T / torch.tensor(NT, dtype=torch.float32)
    sqrt_dt = torch.sqrt(dt)
    num_assets = initial_prices.shape[0]

    # Simulate paths
    Z = torch.randn(Np, NT, num_assets)
    L = torch.linalg.cholesky(rho_matrix)
    Z = Z @ L.T

    Sp = torch.zeros(Np, NT, num_assets, dtype=torch.float32)
    Sp[:, 0, :] = initial_prices

    for t in range(1, NT):
        previous_step = Sp[:, t - 1, :].clone()
        Sp[:, t, :] = previous_step * torch.exp((r - 0.5 * sigmas**2) * dt + sigmas * sqrt_dt * Z[:, t, :])

    # Calculate basket value at maturity
    basket_value = (Sp[:, -1, :] * weights).sum(dim=1)
    payoff = torch.maximum(basket_value - K, torch.tensor(0.0, dtype=torch.float32))

    # Discount payoff to present value
    r_tensor = torch.tensor(r, dtype=torch.float32)
    T_tensor = torch.tensor(T, dtype=torch.float32)
    V = torch.exp(-r_tensor * T_tensor) * payoff.mean()
    return V

# Parameters for testing basket option
initial_prices = torch.tensor([100.0, 82.0, 97.0])
K = 88.0
sigmas = torch.tensor([0.25, 0.3, 0.1])
rho_matrix = torch.tensor([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
weights = torch.tensor([3.0, 1.0, 2.0])
T = 2.0
r = 0.01
Np = 500000
NT = 1000

# Test the basket option pricing
basket_option_value = monte_carlo_basket_option(initial_prices, K, sigmas, rho_matrix, weights, T, r, Np, NT)
print(f"Basket Option Value: {basket_option_value}")
