import torch
from torch.distributions.normal import Normal

def basket_option_price_closed_form(weights, sigmas, initial_prices, r_tensor, T_tensor, K, option_type="call"):
    """
    Compute the price of a basket option under the Bachelier model using the closed-form formula.

    Args:
        weights (torch.Tensor): Weights of the assets in the basket.
        sigmas (torch.Tensor): Volatilities of the assets.
        initial_prices (torch.Tensor): Initial prices of the assets.
        r_tensor (torch.Tensor): Risk-free rate (requires_grad=True).
        T_tensor (torch.Tensor): Time to maturity (requires_grad=True).
        K (float): Strike price.
        option_type (str): "call" or "put".

    Returns:
        torch.Tensor: The price of the basket option.
    """
    # Calculate mean and standard deviation of the basket
    mu_y = torch.sum(weights * (r_tensor * T_tensor + initial_prices))
    sigma_y = torch.sqrt(torch.sum((weights ** 2) * (sigmas ** 2) * T_tensor))

    # Calculate standardized variable x_bar
    x_bar = (K - mu_y) / sigma_y

    # Normal distribution functions
    normal = Normal(0, 1)
    Phi = normal.cdf(-x_bar)  # CDF of the standard normal
    phi = torch.exp(-0.5 * x_bar ** 2) / torch.sqrt(torch.tensor(2.0 * torch.pi))  # PDF of the standard normal

    # Calculate the price using the closed-form formula
    if option_type == "call":
        price = torch.exp(-r_tensor * T_tensor) * (sigma_y * phi - (K - mu_y) * Phi)
    elif option_type == "put":
        price = torch.exp(-r_tensor * T_tensor) * ((mu_y - K) * (1 - Phi) + sigma_y * phi)
    else:
        raise ValueError("Invalid option_type. Choose 'call' or 'put'.")

    return price

def compute_sensitivities(weights, sigmas, initial_prices, r_tensor, T_tensor, K, option_type="call"):
    """
    Compute sensitivities (Greeks) for a basket option using adjoint differentiation.

    Args:
        weights (torch.Tensor): Weights of the assets in the basket.
        sigmas (torch.Tensor): Volatilities of the assets.
        initial_prices (torch.Tensor): Initial prices of the assets.
        r_tensor (torch.Tensor): Risk-free rate (requires_grad=True).
        T_tensor (torch.Tensor): Time to maturity (requires_grad=True).
        K (float): Strike price.
        option_type (str): "call" or "put".

    Returns:
        dict: Sensitivities (delta, vega, rho, theta).
    """
    # Enable gradients for inputs
    initial_prices = initial_prices.clone().detach().requires_grad_(True)
    sigmas = sigmas.clone().detach().requires_grad_(True)

    # Compute the option price
    price = basket_option_price_closed_form(weights, sigmas, initial_prices, r_tensor, T_tensor, K, option_type)

    # Backpropagate to compute gradients
    price.backward()

    # Extract sensitivities
    delta = initial_prices.grad.tolist()  # Sensitivity to initial prices
    vega = sigmas.grad.tolist()  # Sensitivity to volatilities
    rho = r_tensor.grad.item()  # Sensitivity to risk-free rate
    theta = T_tensor.grad.item()  # Sensitivity to time to maturity

    return {
        "delta": delta,
        "vega": vega,
        "rho": rho,
        "theta": theta
    }

# Parameters
sigmas = torch.tensor([0.25, 0.3, 0.1])
initial_prices = torch.tensor([100.0, 82.0, 97.0])
weights = torch.tensor([3.0, 1.0, 2.0])  # Equal weights
r_tensor = torch.tensor(0.01, requires_grad=True)  # Risk-free rate with gradient tracking
T_tensor = torch.tensor(2.0, requires_grad=True)  # Time to maturity with gradient tracking
K = 88.0

# Calculate call and put option prices using the closed-form formula
call_price_closed_form = basket_option_price_closed_form(weights, sigmas, initial_prices, r_tensor, T_tensor, K, option_type="call").item()
put_price_closed_form = basket_option_price_closed_form(weights, sigmas, initial_prices, r_tensor, T_tensor, K, option_type="put").item()

# Compute sensitivities using adjoint differentiation
sensitivities = compute_sensitivities(weights, sigmas, initial_prices, r_tensor, T_tensor, K, option_type="call")

print(f"Call Option Price (Closed Form): {call_price_closed_form:.4f}")
print(f"Put Option Price (Closed Form): {put_price_closed_form:.4f}")
print("Sensitivities:")
for key, value in sensitivities.items():
    if isinstance(value, list):
        print(f"  {key}: {', '.join([f'{v:.4f}' for v in value])}")
    else:
        print(f"  {key}: {value:.4f}")