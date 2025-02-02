
import torch
import os
import sys
sys.path.append(os.path.abspath("/Users/aderdouri/Downloads/EiCNAM/TorchAAD"))

from Methods.base import PricingMethod

import unittest
import torch
from Instruments.european_option import EuropeanOption
from Engine.stochastic_process import LogNormalProcess
#from Methods.monte_carlo import MonteCarloMethod

class MonteCarloMethod(PricingMethod):
    def __init__(self, S0, K, T, r, sigma, num_paths=10000, num_steps=250, device="cpu"):
        """
        Monte Carlo pricing of a European Call option with Greeks using PyTorch.

        Args:
            S0 (float): Initial stock price.
            K (float): Strike price.
            T (float): Time to maturity (years).
            r (float): Risk-free rate.
            sigma (float): Volatility.
            num_paths (int): Number of Monte Carlo paths.
            num_steps (int): Number of time steps.
            device (str): Computation device ('cpu' or 'cuda').
        """
        self.S0 = S0.clone().detach().requires_grad_(True).to(device)
        self.K = K
        self.T = torch.tensor(T, dtype=torch.float32, requires_grad=True, device=device)
        self.r = torch.tensor(r, dtype=torch.float32, requires_grad=True, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, requires_grad=True, device=device)
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T / num_steps
        self.device = device

    def simulate(self):
        """Simulates Geometric Brownian Motion (GBM) paths with gradient tracking only for `S0`."""
        
        # Initialize only the first column with requires_grad=True
        S = torch.zeros((self.num_paths, self.num_steps), dtype=torch.float32, device=self.device)

        # Fill first column manually
        S[:, 0] = self.S0  # Ensure the first column tracks gradients

        dW = torch.randn(self.num_paths, self.num_steps - 1, dtype=torch.float32, device=self.device) * torch.sqrt(
            torch.tensor(self.dt, dtype=torch.float32, device=self.device)
        )

        for t in range(1, self.num_steps):
            S_prev = S[:, t - 1].clone()  # Clone to prevent in-place modification issues
            S[:, t] = S_prev * torch.exp((self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * dW[:, t - 1])

        return S

    def price(self):
        """Computes the Monte Carlo estimate of the option price."""
        S = self.simulate()
        payoffs = torch.maximum(S[:, -1] - self.K, torch.tensor(0.0, device=self.device))
        discount_factor = torch.exp(-self.r * self.T)
        option_price = discount_factor * torch.mean(payoffs)
        return option_price

    def greeks(self):
        """Computes the Greeks (Delta, Gamma, Vega, Rho, Theta) using autograd."""
        option_price = self.price()
        
        # Compute first derivative (Delta)
        option_price.backward(retain_graph=True)  # Compute gradients        
        delta = self.S0.grad  # ∂V / ∂S0        
        vega = self.sigma.grad.item()  # ∂V / ∂σ

        # Compute Rho by taking the derivative with respect to r
        rho = torch.autograd.grad(option_price, self.r, retain_graph=True)[0].item()

        # Compute Theta by taking the derivative with respect to T
        theta = torch.autograd.grad(option_price, self.T, retain_graph=True)[0].item()

        return {
            "Price": option_price.item(),
            "Delta": delta.item(),
            "Vega": vega,
            "Rho": rho,
            "Theta": theta
        }


with torch.autograd.set_detect_anomaly(True):
    # Option Parameters
    S0 = torch.tensor(100.0, dtype=torch.float32, requires_grad=True)
    K = 100         # Strike price
    T = 1.0         # Time to expiration (1 year)
    r = 0.05        # Risk-free rate
    sigma = 0.2     # Volatility

    # Instantiate and compute Greeks
    process = LogNormalProcess(mu=r, sigma=sigma)
    option = MonteCarloMethod(S0, K, T, r, sigma, device="cpu")
    
    # Monte Carlo Simulation
    #process = LogNormalProcess(mu=r, sigma=sigma)
    #mc = MonteCarloMethod(process, S0, T.item(), num_paths, num_steps)
    #paths = mc.simulate()


    results = option.greeks()

    # Print results
    for key, value in results.items():
        print(f"{key}: {value:.5f}")




