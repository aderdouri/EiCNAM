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
        self.mu = mu
        self.sigma = sigma
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
    def evolve(self, S: torch.Tensor, dt: float, dW: torch.Tensor, current_time: float) -> torch.Tensor:
        """
        Evolves the stochastic process using the given S, time step dt, and Brownian motion dW.
        Ensures `mu` and `sigma` are tensors with `requires_grad=True` for gradient tracking.
        
        Args:
            S (torch.Tensor): Current value of the process.
            dt (float): Time step.
            dW (torch.Tensor): Brownian motion increment.
            current_time (float): Current time.

        Returns:
            torch.Tensor: Evolved process value.
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
        self.k = k
        self.nu = nu

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
        drift = self.k * (self.mu - S) * dt
        diffusion = self.nu * torch.sqrt(S) * torch.sqrt(dt) * dW
        return S + drift + diffusion

class CIRPlusPlusProcess(StochasticProcess):
    def __init__(self, mu: float, sigma: float, k: float, theta: float, nu: float, phi, device="cpu"):
        """
        Cox-Ingersoll-Ross (CIR++) Stochastic Process.
        
        Args:
            mu (float): Long-term mean level.
            sigma (float): Volatility coefficient.
            k (float): Speed of mean reversion.
            theta (float): Shift parameter.
            nu (float): Volatility scaling factor.
            phi (callable): Deterministic shift function.
            device (str): Device to perform computations ('cpu' or 'cuda').
        """
        super().__init__(mu, sigma, device)
        self.k = k
        self.theta = theta
        self.nu = nu
        self.phi = phi

    def evolve(self, S: torch.Tensor, dt: float, dW: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evolves the process using the CIR++ dynamics.
        
        Args:
            S (torch.Tensor): Current value of the process.
            dt (float): Time step.
            dW (torch.Tensor): Brownian motion increment.
            t (float): Current time.
        
        Returns:
            torch.Tensor: Evolved process value.
        """
        drift = self.k * (self.mu - S) * dt
        diffusion = self.nu * torch.sqrt(S) * torch.sqrt(dt) * dW
        shift = self.theta * dt + self.phi(t) * dt
        return S + drift + diffusion #+ shift


# def simulate_cir_plus_plus(T, n_simulations, n_steps, k, mu, nu, x0, theta, phi):
#     dt = T / n_steps
#     time_grid = torch.linspace(0, T, n_steps + 1)
#     x = torch.zeros((n_simulations, n_steps + 1))
#     x[:, 0] = x0
#     Z = torch.randn((n_simulations, n_steps))

#     for t in range(1, n_steps + 1):
#         x_t = torch.relu(x[:, t-1])  # Ensure positivity
#         drift = k * (mu - x_t) * dt
#         diffusion = nu * torch.sqrt(x_t) * torch.sqrt(torch.tensor(dt)) * Z[:, t-1]
#         x[:, t] = x_t + drift + diffusion

#     lambda_t = x + theta * dt + phi(time_grid)  # Add deterministic shift
#     return lambda_t, time_grid
