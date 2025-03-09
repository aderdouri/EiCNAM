import torch

class FiniteDifferenceMethod:
    def __init__(self, S_max, T, sigma, r, K, M, N):
        """
        Finite Difference Method (Explicit Scheme for Black-Scholes).

        Args:
            S_max (float): Maximum stock price.
            T (float): Expiry time.
            sigma (float): Volatility.
            r (float): Risk-free rate.
            K (float): Strike price.
            M (int): Number of stock price steps.
            N (int): Number of time steps.
        """
        self.S_max = S_max
        self.T = T
        self.sigma = sigma
        self.r = r
        self.K = K
        self.M = M
        self.N = N
        self.dt = T / N
        self.dS = S_max / M

    def solve(self):
        """
        Solves the Black-Scholes equation using the Explicit Finite Difference Method.

        Returns:
            torch.Tensor: Option prices (shape: [M+1, N+1])
        """
        S = torch.linspace(0, self.S_max, self.M + 1)
        V = torch.maximum(self.K - S, torch.tensor(0.0))  # European Put Payoff

        alpha = 0.5 * self.dt * (self.sigma ** 2 * (torch.arange(self.M) ** 2) - self.r * torch.arange(self.M))
        beta = 1 - self.dt * (self.sigma ** 2 * (torch.arange(self.M) ** 2) + self.r)
        gamma = 0.5 * self.dt * (self.sigma ** 2 * (torch.arange(self.M) ** 2) + self.r * torch.arange(self.M))

        for j in range(self.N - 1, -1, -1):
            V[1:self.M] = alpha[1:self.M] * V[:-2] + beta[1:self.M] * V[1:self.M] + gamma[1:self.M] * V[2:]
            V[-1] = 0  # Boundary condition at S_max
            V[0] = self.K * torch.exp(-self.r * (self.N - j) * self.dt)  # Boundary condition at S=0

        return V
