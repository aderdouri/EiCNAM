class BlackScholesModel:
    """Black-Scholes model for pricing."""
    def __init__(self, r, sigma):
        self.r = r
        self.sigma = sigma

    def simulate(self, S0, T, num_paths, num_steps):
        return f"Simulating paths with S0={S0}, T={T}"
