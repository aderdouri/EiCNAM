import unittest
import torch
from Instruments.european_option import EuropeanOption

from Engine.stochastic_process import NormalProcess, LogNormalProcess
from Engine.simulator import simulate_process

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
        n_paths = 50000  # Number of Monte Carlo paths
        time_steps = 1000  # Number of time steps

        # Generate time grid
        dt = T / time_steps
        time_grid = torch.linspace(0, T, time_steps)

        # Simulate default times using an intensity model
        def simulate_default_times(lambda_t, time_steps, T, n_paths):
            dt = T / time_steps
            cumulative_hazard = torch.cumsum(lambda_t * dt * torch.ones(time_steps), dim=0)
            survival_probs = torch.exp(-cumulative_hazard)
            defaults = torch.rand(n_paths, time_steps) > survival_probs
            default_times = torch.argmax(defaults.type(torch.int), dim=1) * dt
            default_times[default_times == 0] = T + 1  # No default
            return default_times

        # Calculate CVA
        def calculate_cva(LGD, S, K, T, r, lambda_t, time_grid):
            payoffs = torch.maximum(S[:, -1] - K, torch.tensor(0.0))  # Max(S_T - K, 0)
            # Convert r and T to tensors before using them in torch.exp
            discount_factors = torch.exp(-torch.tensor(r, dtype=torch.float32) * torch.tensor(T, dtype=torch.float32))
            cumulative_hazard = torch.sum(lambda_t * dt * torch.ones(time_steps))
            survival_probs = torch.exp(-cumulative_hazard)
            cva = LGD * torch.mean((1 - survival_probs) * discount_factors * payoffs)
            return cva

        # Enable automatic differentiation for lambda_0
        lambda_t = torch.tensor(lambda_0, requires_grad=True)
        process = LogNormalProcess(r, sigma)
        S = simulate_process(process, S0, T, time_steps, n_paths)       

        # Compute CVA using automatic differentiation
        cva = calculate_cva(LGD, S, K, T, r, lambda_t, time_grid)
        
        # Perform backpropagation to compute the gradient of CVA w.r.t. lambda_0
        cva.backward()
        
        # Extract the gradient (delta of CVA with respect to lambda_0)
        delta_cva = lambda_t.grad.item()

        print(f"CVA: {cva.item():.5f}")
        print(f"Delta of CVA w.r.t lambda_0: {delta_cva:.5f}")
                 

if __name__ == '__main__':
    unittest.main()
