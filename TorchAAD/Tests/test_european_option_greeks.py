import unittest
import torch
from Instruments.european_option import EuropeanOption

from Engine.stochastic_process import LogNormalProcess
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
        S0 = torch.tensor(self.option.S0, dtype=torch.float32, requires_grad=True)
        
        K = self.option.K
        T = self.option.T
        r = self.option.r
        sigma = torch.tensor(self.option.sigma, dtype=torch.float32, requires_grad=True)

        num_paths = 5000  # Number of Monte Carlo paths
        num_steps = 1000  # Number of time steps

        with torch.autograd.set_detect_anomaly(True):
            process = LogNormalProcess(r, sigma.item())
            mc = MonteCarloMethod(process, S0.item(), T, num_paths, num_steps)
            paths = mc.simulate()

            # Ensure paths require gradients
            paths = paths.clone().detach().requires_grad_(True)
            payoffs = torch.maximum(paths[:, -1] - K, torch.tensor(0.0))  # Max(S_T - K, 0)
            
            # Convert r and T to tensors before using them in torch.exp
            discount_factors = torch.exp(-torch.tensor(r, dtype=torch.float32) * torch.tensor(T, dtype=torch.float32))
            option_price = torch.mean(discount_factors * payoffs)
            print(f"Option price: {option_price.item():.5f}")    

            # Perform backpropagation to compute the gradient of option_price w.r.t. parameters
            option_price.backward()
            
            # Extract the gradient (delta of CVA with respect to lambda_0)
            delta = S0.grad.item()

            print(f"Delta of option w.r.t S0: {delta:.5f}")
             

if __name__ == '__main__':
    unittest.main()