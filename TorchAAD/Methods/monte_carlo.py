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
        self.dt = T / num_steps

    def simulate(self):
        """
        Runs Monte Carlo simulation using Euler-Maruyama discretization.

        Returns:
            torch.Tensor: Simulated paths (shape: [num_paths, num_steps])
        """
        S = torch.full((self.num_paths, self.num_steps), self.S0, dtype=torch.float32, device=self.process.device)

        # Generate Brownian motion
        dW = torch.randn(self.num_paths, self.num_steps - 1, dtype=torch.float32, 
                         device=self.process.device) * torch.sqrt(
                             torch.tensor(self.dt, dtype=torch.float32, device=self.process.device)
        )

        # Iterate over time steps
        for t in range(1, self.num_steps):
            S[:, t] = self.process.evolve(S[:, t - 1], self.dt, dW[:, t - 1])

        return S

