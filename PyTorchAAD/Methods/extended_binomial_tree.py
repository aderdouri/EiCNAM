import numpy as np
import torch
from Methods.binomial_tree import BinomialTreeMethod

class ExtendedBinomialTreeMethod(BinomialTreeMethod):
    def __init__(self, num_steps, exercise_times=None):
        super().__init__(num_steps, exercise_times)

    def price(self, S0, K, T, r, sigma, exercise_dates, is_call=True):
        """
        Price a Bermudan option using an extended binomial tree with PyTorch.
        
        Parameters:
        - S0: Initial stock price (float)
        - K: Strike price (float)
        - T: Time to maturity (float, in years)
        - r: Risk-free rate (float)
        - sigma: Volatility (float)
        - exercise_dates: List of exercise step indices (e.g., [3, 6, 9, 12])
        - is_call: Whether the option is a call (default True). False for a put option.
        
        Returns:
        - option_price: float
        """
        S0 = torch.tensor(S0, requires_grad=True)
        dt = torch.tensor(T / self.num_steps)  # Time step
        u = torch.exp(torch.tensor(sigma) * torch.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (torch.exp(torch.tensor(r) * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize asset price tree
        S = torch.zeros((self.num_steps + 1, self.num_steps + 1))
        S[0, 0] = S0
        for i in range(1, self.num_steps + 1):
            S[i, :i] = S[i - 1, :i] * u
            S[i, 1:i + 1] = S[i - 1, :i] * d
        
        # Initialize option value tree
        option_values = torch.zeros_like(S)
        payoff = torch.maximum(S[:, :self.num_steps + 1] - K, torch.tensor(0.0)) if is_call else \
                 torch.maximum(K - S[:, :self.num_steps + 1], torch.tensor(0.0))
        
        # Backward induction
        for i in range(self.num_steps - 1, -1, -1):  
            if i in exercise_dates:  # Bermudan exercise
                option_values[i, :i + 1] = torch.maximum(payoff[i, :i + 1],
                                                         torch.exp(-torch.tensor(r) * dt) * (p * option_values[i + 1, :i + 1] +
                                                                                             (1 - p) * option_values[i + 1, 1:i + 2]))
            else:  # No early exercise
                option_values[i, :i + 1] = torch.exp(-torch.tensor(r) * dt) * (p * option_values[i + 1, :i + 1] +
                                                                               (1 - p) * option_values[i + 1, 1:i + 2])

        option_price = option_values[0, 0]
        
        return option_price
    

    def calculate_greeks(self, S0, K, T, r, sigma, exercise_dates, is_call=True):
        """
        Calculate sensitivities (Delta, Vega, Rho, Theta) using automatic differentiation.

        Args:
            S0: Initial asset price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            exercise_dates: List of exercise step indices.
            is_call: Whether the option is a call (default True). False for a put option.

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
            option_price = self.price(S0_t, K, T_t, r_t, sigma_t, exercise_dates, is_call)

            # Ensure gradients are calculated
            option_price.backward(retain_graph=True)

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
