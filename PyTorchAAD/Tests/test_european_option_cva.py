import unittest
import torch
import numpy as np
from Instruments.european_option import EuropeanOption
from Engine.stochastic_process import LogNormalProcess, IntensityProcess
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

    def test_cva_calculation(self):
        # Use inputs from self.option
        S0 = self.option.S0
        K = self.option.K
        T = self.option.T
        r = self.option.r
        sigma = self.option.sigma

        # CVA and intensity model parameters
        LGD = 0.6  # Loss given default
        lambda_0 = 1.0  # Initial hazard rate
        num_paths = 50  # Number of Monte Carlo paths
        num_steps = 1000  # Number of time steps

        # Generate time grid
        dt = T / num_steps

        # Calculate CVA
        def calculate_cva(LGD, S, K, T, r, lambda_t):
            payoffs = torch.maximum(S[:, -1] - K, torch.tensor(0.0))  # Max(S_T - K, 0)
            discount_factors = torch.exp(-torch.tensor(r, dtype=torch.float32) * T)
            cumulative_hazard = torch.sum(lambda_t * dt * torch.ones(num_steps))
            survival_probs = torch.exp(-cumulative_hazard)
            cva = LGD * torch.mean((1 - survival_probs) * discount_factors * payoffs)
            return cva

        # Enable automatic differentiation for lambda_0
        lambda_t = torch.tensor(lambda_0, requires_grad=True)

        process = LogNormalProcess(r, sigma)
        mc = MonteCarloMethod(process, S0, T, num_paths, num_steps)        
        paths = mc.simulate()

        # Compute CVA using automatic differentiation
        cva = calculate_cva(LGD, paths, K, T, r, lambda_t)
        
        # Perform backpropagation to compute the gradient of CVA w.r.t. lambda_0
        cva.backward()
        
        # Extract the gradient (delta of CVA with respect to lambda_0)
        delta_cva = lambda_t.grad.item()

        print(f"CVA: {cva.item():.5f}")
        print(f"Delta of CVA w.r.t lambda_0: {delta_cva:.5f}")

        # Add assertions to validate the results
        self.assertTrue(cva.item() > 0)
        self.assertTrue(delta_cva != 0)

if __name__ == '__main__':
    unittest.main()
