
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

    def test_cva_closed_form_calculation(self):
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
        
        # Given constants
        lambda0 = 1.0
        LGD = 0.6

        # Convert constants to PyTorch tensors
        k = torch.tensor(0.5, dtype=torch.float64)
        mu = torch.tensor(1.0, dtype=torch.float64)
        nu = torch.tensor(0.25, dtype=torch.float64)

        # Compute h
        h = torch.sqrt(k**2 + 2 * nu**2)

        # Compute A(0, T)
        numerator_A = 2 * h * torch.exp((k + h) * T / 2)
        denominator_A = 2 * h + (k + h) * (torch.exp(h * T) - 1)
        exponent_A = (2 * k * mu) / (nu**2)
        A_0_T = (numerator_A / denominator_A)**exponent_A

        # Compute B(0, T)
        numerator_B = 2 * h * (torch.exp(h * T) - 1)
        denominator_B = 2 * h + (k + h) * (torch.exp(h * T) - 1)
        B_0_T = numerator_B / denominator_B

        # Print results
        print(f"h: {h.item()}")
        print(f"A(0, T): {A_0_T.item()}")
        print(f"B(0, T): {B_0_T.item()}")

        CVA = LGD * (1 - A_0_T * torch.exp(-lambda0 * B_0_T)) * option_price    

        print(f"CVA: {CVA.item()}")

        # Compute CVA sensitivity with respect to λ0
        p_bar = 1.0
        u_bar = LGD * option_price * p_bar 
        lambda0_bar = A_0_T * B_0_T * torch.exp(-B_0_T * lambda0) * u_bar

        print(f"CVA Sensitivity with respect to λ0: {lambda0_bar.item()}")


if __name__ == '__main__':
    unittest.main()