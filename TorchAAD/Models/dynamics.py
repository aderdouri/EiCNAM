# dynamics.py
import torch

class HestonProcess:
    def __init__(self, v0, mu, kappa, theta, sigma, rho, device="cpu"):
        self.v0 = v0  # Initial variance
        self.mu = mu  # Drift
        self.kappa = kappa  # Mean-reversion speed
        self.theta = theta  # Long-term variance
        self.sigma = sigma  # Volatility of volatility
        self.rho = rho  # Correlation
        self.device = device

    def evolve(self, S, v, dt, dW_S, dW_v):
        v_next = torch.clamp(v + self.kappa * (self.theta - v) * dt + self.sigma * torch.sqrt(v) * dW_v, min=0)
        S_next = S * torch.exp((self.mu - 0.5 * v) * dt + torch.sqrt(v) * dW_S)
        return S_next, v_next
