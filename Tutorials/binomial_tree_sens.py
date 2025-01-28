import torch

class ExtendedBinomialTree:
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def price_bermudan_option(self, S0, K, T, r, sigma, exercise_dates, is_call=True):
        """
        Price a Bermudan option using an extended binomial tree.
        
        Parameters:
        - S0: Initial stock price (float or tensor)
        - K: Strike price (float)
        - T: Time to maturity (float, in years)
        - r: Risk-free rate (float)
        - sigma: Volatility (float)
        - exercise_dates: List of exercise step indices (e.g., [3, 6, 9, 12])
        - is_call: Whether the option is a call (default True). False for a put option.
        
        Returns:
        - option_price: PyTorch tensor (float)
        """
        dt = T / self.num_steps  # Time step
        u = torch.exp(sigma * torch.sqrt(torch.tensor(dt)))  # Up factor
        d = 1 / u  # Down factor
        # Convert r and dt to tensors before performing operations
        p = (torch.exp(torch.tensor(r * dt)) - d) / (u - d)  # Risk-neutral probability

        # Initialize asset price tree
        S = torch.zeros((self.num_steps + 1, self.num_steps + 1))
        S[0, 0] = S0
        for i in range(1, self.num_steps + 1):
            S[i, :i] = S[i - 1, :i] * u
            S[i, 1:i + 1] = S[i - 1, :i].clone() * d  # Avoid in-place modification
        
        # Initialize option value tree
        option_values = torch.zeros_like(S)
        payoff = torch.maximum(S[:, :self.num_steps + 1] - K, torch.tensor(0.0)) if is_call else \
                 torch.maximum(K - S[:, :self.num_steps + 1], torch.tensor(0.0))
        
        # Backward induction
        # The loop iterates from num_steps -1  down to 0 inclusive
        for i in range(self.num_steps - 1, -1, -1):  
            if i in exercise_dates:  # Bermudan exercise
                # Convert -r * dt to a tensor
                # Slicing is adjusted to stay within bounds
                option_values[i, :i + 1] = torch.maximum(payoff[i, :i + 1],
                                                         torch.exp(torch.tensor(-r * dt)) * (p * option_values[i + 1, :i + 1].clone() +
                                                                               (1 - p) * option_values[i + 1, 1:i + 2].clone()))
            else:  # No early exercise
                # Convert -r * dt to a tensor
                # Slicing is adjusted to stay within bounds
                option_values[i, :i + 1] = torch.exp(torch.tensor(-r * dt)) * (p * option_values[i + 1, :i + 1].clone() +
                                                                 (1 - p) * option_values[i + 1, 1:i + 2].clone())

        return option_values[0, 0]

    def calculate_greeks(self, S0, K, T, r, sigma, exercise_dates, is_call=True):
        """
        Calculate Greeks (Delta, Gamma, Vega) using automatic differentiation.

        Parameters:
        Same as `price_bermudan_option`.

        Returns:
        - greeks: Dictionary containing Delta, Gamma, and Vega.
        """
        # Convert S0 and sigma to float tensors before setting requires_grad
        S0 = torch.tensor(S0, dtype=torch.float32, requires_grad=True) 
        sigma = torch.tensor(sigma, dtype=torch.float32, requires_grad=True)

        # Calculate option price
        price = self.price_bermudan_option(S0, K, T, r, sigma, exercise_dates, is_call)

        # Compute Delta (∂Price/∂S0)
        price.backward(retain_graph=True)
        delta = S0.grad.item()

        # Compute Gamma (∂²Price/∂S0²)
        S0.grad = None  # Resetting gradient before second backward pass
        price = self.price_bermudan_option(S0, K, T, r, sigma, exercise_dates, is_call)
        price.backward(retain_graph=True)  # Recomputing gradients
        gamma = S0.grad.item()

        # Compute Vega (∂Price/∂σ)
        sigma.grad = None  # Resetting gradient
        price = self.price_bermudan_option(S0, K, T, r, sigma, exercise_dates, is_call)
        price.backward()
        vega = sigma.grad.item()

        return {"Delta": delta, "Gamma": gamma, "Vega": vega}


# Example Usage
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # Parameters
    S0 = 100  # Initial stock price
    K = 100  # Strike price
    T = 1  # Maturity (in years)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    num_steps = 50  # Number of steps in the binomial tree
    exercise_dates = [10, 20, 30, 40, 50]  # Bermudan exercise points

    # Initialize the extended binomial tree
    tree = ExtendedBinomialTree(num_steps)

    # Price the option
    option_price = tree.price_bermudan_option(S0, K, T, r, sigma, exercise_dates, is_call=True)
    print(f"Bermudan Call Option Price: {option_price.item()}")

    # Calculate Greeks
    greeks = tree.calculate_greeks(S0, K, T, r, sigma, exercise_dates, is_call=True)
    print(f"Greeks: {greeks}")