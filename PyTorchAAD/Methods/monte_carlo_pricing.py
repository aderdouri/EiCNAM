import torch
from Methods.base import PricingMethod

class MonteCarloPricing(PricingMethod):
    def __init__(self, num_paths, num_steps):
        self.num_paths = num_paths
        self.num_steps = num_steps

    def simulate_asset_paths(self, S0, T, r, sigma):
        dt = torch.tensor(T / self.num_steps)  # Convert dt to tensor
        paths = torch.zeros((self.num_steps + 1, self.num_paths))
        paths[0] = S0

        for t in range(1, self.num_steps + 1):
            z = torch.randn(self.num_paths)
            paths[t] = paths[t - 1] * torch.exp((r - 0.5 * sigma ** 2) * dt + sigma * torch.sqrt(dt) * z)

        return paths

    def price_best_of_two_assets_bermudan_option(self, S0_1, S0_2, K, T, r, sigma_1, sigma_2, rho, exercise_dates, is_call=True):
        dt = torch.tensor(T / self.num_steps)  # Convert dt to tensor
        discount_factor = torch.exp(-r * dt)

        # Simulate asset paths
        paths_1 = self.simulate_asset_paths(S0_1, T, r, sigma_1)
        paths_2 = self.simulate_asset_paths(S0_2, T, r, sigma_2)

        # Initialize option values
        option_values = torch.zeros((self.num_steps + 1, self.num_paths))

        # Calculate payoff at maturity
        if is_call:
            payoff = torch.maximum(torch.maximum(paths_1[-1], paths_2[-1]) - K, torch.tensor(0.0))
        else:
            payoff = torch.maximum(K - torch.maximum(paths_1[-1], paths_2[-1]), torch.tensor(0.0))

        option_values[-1] = payoff

        # Backward induction
        for t in range(self.num_steps - 1, -1, -1):
            if t in exercise_dates:
                if is_call:
                    payoff = torch.maximum(torch.maximum(paths_1[t], paths_2[t]) - K, torch.tensor(0.0))
                else:
                    payoff = torch.maximum(K - torch.maximum(paths_1[t], paths_2[t]), torch.tensor(0.0))

                continuation_value = discount_factor * option_values[t + 1]
                option_values[t] = torch.maximum(payoff, continuation_value)
            else:
                option_values[t] = discount_factor * option_values[t + 1]

        option_price = option_values[0].mean()
        return option_price
