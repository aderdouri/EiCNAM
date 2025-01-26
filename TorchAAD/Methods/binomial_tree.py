import torch
from TorchAAD.Methods.base import PricingMethod

class BinomialTreeMethod(PricingMethod):
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def price(self, instrument):
        return f"Binomial Tree price for {instrument.name} with S0={instrument.S0}"

    def price_bermudan_option(self, instrument, exercise_dates):
        S0 = instrument.S0
        K = instrument.strike
        T = instrument.maturity
        r = instrument.rate
        sigma = instrument.volatility
        dt = torch.tensor(T / self.num_steps)
        u = torch.exp(sigma * torch.sqrt(dt))
        d = 1 / u
        q = (torch.exp(r * dt) - d) / (u - d)

        # Initialize asset prices at maturity
        asset_prices = S0 * d**torch.arange(self.num_steps, -1, -1) * u**torch.arange(0, self.num_steps + 1)
        option_values = torch.maximum(torch.zeros_like(asset_prices), K - asset_prices)

        # Step back through the tree
        for step in range(self.num_steps - 1, -1, -1):
            option_values = torch.exp(-r * dt) * (q * option_values[1:] + (1 - q) * option_values[:-1])
            if step in exercise_dates:
                asset_prices = S0 * d**torch.arange(step, -1, -1) * u**torch.arange(0, step + 1)
                option_values = torch.maximum(option_values, K - asset_prices)

        return option_values[0]

    def calculate_cva(self, instrument, default_prob, recovery_rate):
        dt = torch.tensor(instrument.maturity / self.num_steps)
        u = torch.exp(instrument.volatility * torch.sqrt(dt))
        d = 1 / u
        p = (torch.exp(instrument.rate * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        asset_prices = torch.zeros(self.num_steps + 1)
        asset_prices[0] = instrument.S0 * (d ** self.num_steps)
        for i in range(1, self.num_steps + 1):
            asset_prices[i] = asset_prices[i - 1].clone() * (u / d)
        
        # Initialize option values at maturity
        option_values = torch.maximum(asset_prices - instrument.strike, torch.tensor(0.0))
        
        # Backward induction to get option value at t=0
        for step in range(self.num_steps - 1, -1, -1):
            for i in range(step + 1):
                option_values[i] = torch.exp(-instrument.rate * dt) * (p * option_values[i + 1] + (1 - p) * option_values[i])
        
        # Calculate CVA manually
        cva = torch.tensor(0.0)
        for step in range(1, self.num_steps + 1):
            default_leg = (1 - recovery_rate) * torch.exp(-instrument.rate * step * dt) * option_values[step]
            survival_prob = (1 - default_prob) ** step
            cva += default_leg * (1 - survival_prob)
        
        return cva
