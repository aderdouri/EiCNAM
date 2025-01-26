from .base import PricingMethod

class MonteCarloMethod(PricingMethod):
    def __init__(self, num_paths, num_steps):
        self.num_paths = num_paths
        self.num_steps = num_steps

    def price(self, instrument):
        if instrument.name == "European Option":
            return f"Monte Carlo price for {instrument.name} with S0={instrument.S0}"
        elif instrument.name == "Bermudan Option":
            return f"Monte Carlo price for {instrument.name} with S0={instrument.S0}"
