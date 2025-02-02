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
        """ Test European Option pricing using Monte Carlo simulation with gradient tracking """
        # Convert option parameters to PyTorch tensors with requires_grad for differentiation
        S0 = torch.tensor(self.option.S0, dtype=torch.float32, requires_grad=True)        
        K = self.option.K
        T = torch.tensor(self.option.T, dtype=torch.float32, requires_grad=True)  # Time doesn't need gradients
        r = torch.tensor(self.option.r, dtype=torch.float32, requires_grad=True)  # Risk-free rate is fixed
        sigma = torch.tensor(self.option.sigma, dtype=torch.float32, requires_grad=True)  # Track volatility sensitivity

        num_paths = 50  # Number of Monte Carlo paths
        num_steps = 1000  # Number of time steps

        with torch.autograd.set_detect_anomaly(True):
            # Ensure process parameters are differentiable
            process = LogNormalProcess(r, sigma)
            process.r = r
            process.sigma = sigma

            # Monte Carlo Simulation
            mc = MonteCarloMethod(process, S0, T, num_paths, num_steps)
            paths = mc.simulate()

            # Compute Payoffs: European Call (max(S_T - K, 0))
            payoffs = torch.maximum(paths[:, -1] - K, torch.tensor(0.0, dtype=torch.float32, device=paths.device))
            
            # Compute discounted expected value (risk-neutral pricing)
            discount_factor = torch.exp(-r * T)
            option_price = torch.mean(discount_factor * payoffs)

            print(f"Option price: {option_price.item():.5f}")    

            # Perform backpropagation to compute option Greeks
            option_price.backward(retain_graph=True)  # Ensures gradients are retained

            # Extract Delta (dV/dS0) and Vega (dV/dσ)
            delta = S0.grad.item()
            vega = sigma.grad.item()

            # Compute Rho (dV/dr) by taking the derivative with respect to r
            rho = torch.autograd.grad(option_price, r, retain_graph=True)[0].item()

            # Compute Theta (dV/dT) by taking the derivative with respect to T
            theta = torch.autograd.grad(option_price, T, retain_graph=True)[0].item()

            print(f"Delta (dV/dS0): {delta:.5f}")
            print(f"Vega (dV/dσ): {vega:.5f}")
            print(f"Rho (dV/dr): {rho:.5f}")
            print(f"Theta (dV/dT): {theta:.5f}")


if __name__ == '__main__':
    unittest.main()
