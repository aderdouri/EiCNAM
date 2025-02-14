import torch

# Simulate stock price paths
def simulate_stock_price(S0, r, T, sigma, n_simulations, n_steps):
    dt = T / n_steps
    Z = torch.randn((n_simulations, n_steps))  # Standard normal random variables
    stock_paths = torch.zeros((n_simulations, n_steps + 1))
    stock_paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        stock_paths[:, t] = stock_paths[:, t-1] * torch.exp(
            (r - 0.5 * sigma**2) * dt + sigma * torch.sqrt(torch.tensor(dt)) * Z[:, t-1]
        )
    return stock_paths

# Simulate CIR++ default intensity
def simulate_cir_plus_plus(T, n_simulations, n_steps, k, mu, nu, x0, phi):
    dt = T / n_steps
    time_grid = torch.linspace(0, T, n_steps + 1)
    x = torch.zeros((n_simulations, n_steps + 1))
    x[:, 0] = x0
    Z = torch.randn((n_simulations, n_steps))

    for t in range(1, n_steps + 1):
        x_t = torch.relu(x[:, t-1])  # Ensure positivity
        drift = k * (mu - x_t) * dt
        diffusion = nu * torch.sqrt(x_t) * torch.sqrt(torch.tensor(dt)) * Z[:, t-1]
        x[:, t] = x_t + drift + diffusion

    lambda_t = x + phi(time_grid)  # Add deterministic shift
    return lambda_t, time_grid

# Modified calculate_cva to support autograd
def calculate_cva_autograd(S0, r, T, sigma, K, LGD, k, mu, nu, x0, phi, n_simulations, n_steps):
    # Ensure x0 is a tensor that requires gradients
    x0 = torch.tensor(x0, requires_grad=True)

    # Simulate stock paths
    stock_paths = simulate_stock_price(S0, r, T, sigma, n_simulations, n_steps)
    print(f"stock_paths : {stock_paths.shape}")


    # Simulate CIR++ paths
    lambda_t, time_grid = simulate_cir_plus_plus(T, n_simulations, n_steps, k, mu, nu, x0, phi)

    print(f"lambda_t : {lambda_t[:, -1].detach().numpy()}")

    # Compute integrated hazard and survival probabilities
    integrated_hazard = torch.cumsum(lambda_t[:, :-1] * dt, dim=1)
    print(f"Average integrated hazard: {torch.mean(integrated_hazard).item():.4f}")
    survival_prob = torch.exp(-integrated_hazard[:, -1])  # Survival probability at T
    print(f"Average survival probability: {torch.mean(survival_prob).item():.4f}")

    # Compute payoff and CVA
    ST = stock_paths[:, -1]  # Terminal stock prices
    payoff = torch.relu(ST - K)  # Call option payoff
    discounted_payoff = torch.exp(torch.tensor(-r * T)) * payoff
    cva = LGD * torch.mean((1 - survival_prob) * discounted_payoff)

    return cva, x0

def calculate_european_call_price(S0, r, T, sigma, K, n_simulations, n_steps):
    """
    Calculate the price of a European call option using Monte Carlo simulation.
    
    Args:
        S0 (float): Initial stock price.
        r (float): Risk-free rate.
        T (float): Time to maturity.
        sigma (float): Volatility.
        K (float): Strike price.
        n_simulations (int): Number of simulations.
        n_steps (int): Number of time steps.
    
    Returns:
        float: Estimated price of the European call option.
    """
    stock_paths = simulate_stock_price(S0, r, T, sigma, n_simulations, n_steps)
    ST = stock_paths[:, -1]  # Terminal stock prices
    payoff = torch.relu(ST - K)  # Call option payoff
    discounted_payoff = torch.exp(torch.tensor(-r * T)) * payoff
    price = torch.mean(discounted_payoff).item()
    
    return price

# Parameters for stock simulation

S0 = 100.0  # Initial stock price
r = 0.01    # Risk-free rate
T = 2.0     # Time to maturity
sigma = 0.25  # Volatility
K = 90.0    # Strike price

# λ0 = 1.0, k= 0.5, µ= 1.0, ν = 0.25 and a LGD = 0.6.
# Parameters for CIR++ model
k = 0.5      # Speed of mean reversion
mu = 1.0    # Long-term mean level
nu = 0.25     # Volatility of the CIR process
x0 = 1.0    # # Initial default intensity (lambda_0)
LGD = 0.6    # Loss Given Default

phi = lambda t: 0.005 * torch.exp(-0.1 * t)  # Deterministic shift

# Simulation parameters
n_simulations = 100000
n_steps = 365
dt = T / n_steps

# Calculate European call option price
price = calculate_european_call_price(S0, r, T, sigma, K, n_simulations, n_steps)
print(f"European Call Option Price: {price:.4f}")

# Calculate CVA and differentiate w.r.t. x0
cva, x0_tensor = calculate_cva_autograd(S0, r, T, sigma, K, LGD, k, mu, nu, x0, phi, 
                                        n_simulations, n_steps)

# Perform backward pass to compute gradient
cva.backward()

# Sensitivity is the gradient of CVA w.r.t. x0
sensitivity = x0_tensor.grad.item()

# Output results
print(f"CVA: {cva.item():.4f}")
print(f"Sensitivity of CVA to Initial Lambda: {sensitivity:.4f}")
