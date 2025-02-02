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
    # def evolve(self, S: torch.Tensor, dt: float, dW: torch.Tensor) -> torch.Tensor:
    #     """
    #     Evolves the process using Geometric Brownian Motion (GBM) dynamics.
        
    #     Args:
    #         S (torch.Tensor): Current value of the process.
    #         dt (float): Time step.
    #         dW (torch.Tensor): Brownian motion increment.

    #     Returns:
    #         torch.Tensor: Evolved process value.
    #     """
    #     return S * torch.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW)


    def evolve(self, S: torch.Tensor, dt: float, dW: torch.Tensor) -> torch.Tensor:
        """
        Evolves the stochastic process using the given S, time step dt, and Brownian motion dW.
        Ensures `mu` and `sigma` are tensors with `requires_grad=True` for gradient tracking.
        """
        #mu = torch.tensor(self.mu, dtype=torch.float32, device=S.device, requires_grad=True)
        #sigma = torch.tensor(self.sigma, dtype=torch.float32, device=S.device, requires_grad=True)

        return S * torch.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW)

