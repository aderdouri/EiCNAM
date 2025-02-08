import torch

class StochasticProcess:
    def __init__(self, mu: float, sigma: float, device="cpu"):
        """
        Base class for stochastic processes.
        
        Args:
            mu (float): Drift coefficient.
            sigma (float): Volatility coefficient.
            device (str): Device to perform computations ('cpu' or 'cuda').
        """
        self.mu = mu #torch.tensor(mu, dtype=torch.float32, device=device)
        self.sigma = sigma #torch.tensor(sigma, dtype=torch.float32, device=device)
        self.device = device

    def evolve(self, S: torch.Tensor, dt: float, dW: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to evolve the process. Must be implemented by subclasses.
        
        Args:
            S (torch.Tensor): Current value of the process.
            dt (float): Time step.
            dW (torch.Tensor): Brownian motion increment.

        Returns:
            torch.Tensor: Evolved process value.
        """
        raise NotImplementedError("The evolve method must be implemented in a subclass.")

class NormalProcess(StochasticProcess):
    def evolve(self, S: torch.Tensor, dt: float, dW: torch.Tensor) -> torch.Tensor:
        """
        Evolves the process using Arithmetic Brownian Motion (ABM) dynamics.
        
        Args:
            S (torch.Tensor): Current value of the process.
            dt (float): Time step.
            dW (torch.Tensor): Brownian motion increment.

        Returns:
            torch.Tensor: Evolved process value.
        """
        return S + self.mu * dt + self.sigma * dW

class LogNormalProcess(StochasticProcess):
    def evolve(self, S: torch.Tensor, dt: float, dW: torch.Tensor) -> torch.Tensor:
        """
        Evolves the stochastic process using the given S, time step dt, and Brownian motion dW.
        Ensures `mu` and `sigma` are tensors with `requires_grad=True` for gradient tracking.
        """
        return S * torch.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW)

class IntensityProcess(StochasticProcess):
    def __init__(self, mu: float, sigma: float, k: float, nu: float, device="cpu"):
        """
        Mean-Reverting Stochastic Process.
        
        Args:
            mu (float): Mean-reversion level.
            sigma (float): Initial volatility coefficient (unused in this process, for compatibility).
            k (float): Speed of mean reversion.
            nu (float): Volatility scaling factor.
            device (str): Device to perform computations ('cpu' or 'cuda').
        """
        super().__init__(mu, sigma, device)
        self.k = torch.tensor(k, dtype=torch.float32, device=self.device)
        self.nu = torch.tensor(nu, dtype=torch.float32, device=self.device)

    def evolve(self, S: torch.Tensor, dt: float, dW: torch.Tensor) -> torch.Tensor:
        """
        Evolves the process using the given dynamics.
        
        Args:
            S (torch.Tensor): Current value of the process.
            dt (float): Time step (h).
            dW (torch.Tensor): Standard normal random variable (Z_i).
        
        Returns:
            torch.Tensor: Evolved process value.
        """
        dt = torch.tensor(dt, dtype=torch.float32, device=S.device)
        drift = self.k * (self.mu - S) * dt
        diffusion = self.nu * torch.sqrt(S) * torch.sqrt(dt) * dW
        return S + drift + diffusion
