from .base import PricingMethod
import torch
class MonteCarloMethod(PricingMethod):
    def __init__(self, num_paths, num_steps):
        self.num_paths = num_paths
        self.num_steps = num_steps

    def __init__(self, process, S0, T, num_steps, num_paths):
        """
        Monte Carlo Simulation for stochastic processes.

        Args:
            process: A subclass of StochasticProcess.
            S0: Initial value of the process.
            T: Total time.
            steps: Number of time steps.
            n_paths: Number of Monte Carlo paths.
        """
        self.process = process
        self.S0 = S0
        self.T = T
        self.num_paths = num_paths
        self.num_steps = num_steps
        if not torch.is_tensor(T):
            T = torch.tensor(T, dtype=torch.float32)
        self.dt = T / num_steps

    def simulate(self):
        """
        Runs Monte Carlo simulation using Euler-Maruyama discretization.

        Returns:
            torch.Tensor: Simulated paths (shape: [num_paths, num_steps])
        """
        S = torch.zeros((self.num_paths, self.num_steps), dtype=torch.float32)

        # Fill first column manually
        S[:, 0] = self.S0  # Ensure the first column tracks gradients

        # Generate Brownian motion
        dW = torch.randn(self.num_paths, self.num_steps - 1, dtype=torch.float32) * torch.sqrt(self.dt)

        # Iterate over time steps
        for t in range(1, self.num_steps):
            S_prev = S[:, t - 1].clone()  # Clone to prevent in-place modification issues
            S[:, t] = self.process.evolve(S_prev, self.dt, dW[:, t - 1])  # Clone to prevent in-place errors

        return S            

