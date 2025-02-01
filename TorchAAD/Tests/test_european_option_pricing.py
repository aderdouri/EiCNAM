
import unittest
import torch
from Instruments.european_option import EuropeanOption

from Engine.stochastic_process import LogNormalProcess
from Engine.simulator import simulate_process
from Methods.monte_carlo import MonteCarloMethod

class TestEuropeanOption(unittest.TestCase):
    def setUp(self):        
        self.option = EuropeanOption(S0=100.0, K=90.0, T=2.0, r=0.01, sigma=0.25)

    def test_initialization(self):
        self.assertEqual(self.option.S0, 100.0)
        self.assertEqual(self.option.K, 90.0)
        self.assertEqual(self.option.T, 2.0)
        self.assertEqual(self.option.r, 0.01)
        self.assertEqual(self.option.sigma, 0.25)

    def test_pricing(self):
        # Use inputs from self.option
        S0 = self.option.S0
        K = self.option.K
        T = self.option.T
        r = self.option.r
        sigma = self.option.sigma

        num_paths = 500000  # Number of Monte Carlo paths
        num_steps = 1000  # Number of time steps

        process = LogNormalProcess(r, sigma)
        mc = MonteCarloMethod(process, S0, T, num_paths, num_steps)
        paths = mc.simulate()

        payoffs = torch.maximum(paths[:, -1] - K, torch.tensor(0.0))  # Max(S_T - K, 0)
        # Convert r and T to tensors before using them in torch.exp
        discount_factors = torch.exp(-torch.tensor(r, dtype=torch.float32) * torch.tensor(T, dtype=torch.float32))
        option_price = torch.mean(discount_factors * payoffs)
        print(f"Option price: {option_price.item():.5f}")                 

if __name__ == '__main__':
    unittest.main()