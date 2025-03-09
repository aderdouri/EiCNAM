import torch
from Methods.base import PricingMethod

class BinomialTreeMethod(PricingMethod):
    def __init__(self, num_steps, exercise_times=None):
        self.num_steps = num_steps
        self.exercise_times = exercise_times if exercise_times is not None else []

    def price(self, instrument, S0=None, T=None, r=None, sigma=None):
        S0 = S0 if S0 is not None else instrument.S0
        K = instrument.strike
        T = T if T is not None else instrument.maturity
        r = r if r is not None else instrument.rate
        sigma = sigma if sigma is not None else instrument.volatility
        dt = T / torch.tensor(self.num_steps, dtype=torch.float32)  # Ensure dt is a tensor        
        u = torch.exp(sigma * torch.sqrt(dt))
        d = 1 / u
        q = (torch.exp(r * dt) - d) / (u - d)

        # Initialize asset prices at maturity
        asset_prices = S0 * d**torch.arange(self.num_steps, -1, -1) * u**torch.arange(0, self.num_steps + 1)
        option_values = torch.maximum(torch.zeros_like(asset_prices), K - asset_prices)

        # Step back through the tree
        for step in range(self.num_steps - 1, -1, -1):
            option_values = torch.exp(-r * dt) * (q * option_values[1:] + (1 - q) * option_values[:-1])
            if (step + 1) * dt.item() in self.exercise_times:
                option_values = torch.maximum(option_values, K - asset_prices[:step + 1])

        return option_values[0]

    def calculate_greeks(self, instrument):
        S0 = torch.tensor(instrument.S0, dtype=torch.float32, requires_grad=True)
        T = torch.tensor(instrument.maturity, dtype=torch.float32, requires_grad=True)
        r = torch.tensor(instrument.rate, dtype=torch.float32, requires_grad=True)
        sigma = torch.tensor(instrument.volatility, dtype=torch.float32, requires_grad=True)

        price = self.price(instrument, S0=S0, T=T, r=r, sigma=sigma)
        price.backward()

        delta = S0.grad.item()
        vega = sigma.grad.item()
        rho = r.grad.item()
        theta = T.grad.item()

        return {
            'price': price.item(),
            'delta': delta,
            'vega': vega,
            'rho': rho,
            'theta': theta
        }
