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
        return
        # Use inputs from self.option
        S0 = self.option.S0
        K = self.option.K
        #T = torch.tensor(self.option.T, dtype=torch.float32, requires_grad=True)
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
        time_grid = torch.linspace(0, T, num_steps)

        # Calculate CVA
        def calculate_cva(LGD, S, K, T, r, lambda_t, time_grid):
            payoffs = torch.maximum(S[:, -1] - K, torch.tensor(0.0))  # Max(S_T - K, 0)
            # Convert r and T to tensors before using them in torch.exp
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
        cva = calculate_cva(LGD, paths, K, T, r, lambda_t, time_grid)
        
        # Perform backpropagation to compute the gradient of CVA w.r.t. lambda_0
        cva.backward()
        
        # Extract the gradient (delta of CVA with respect to lambda_0)
        delta_cva = lambda_t.grad.item()

        print(f"CVA: {cva.item():.5f}")
        print(f"Delta of CVA w.r.t lambda_0: {delta_cva:.5f}")

    def test_intensity_process(self):
        process = IntensityProcess(mu=1.0, sigma=0.0, k=0.5, nu=0.2)
        # Initialize process
        S0 = torch.tensor(1.0)  # Initial value
        dt = 1 / 252            # Daily time step
        dW = torch.randn(1)     # Random normal increment
        S_next = process.evolve(S0, dt, dW)
        #print(S_next)
                 

    def test_cir_initensity_cva(self):
        lambda_0 = 1.0  # Initial hazard rate
        k = 0.5
        mu = 1.0
        nu = 0.25
        LGD = 0.6  # Loss given default
        num_paths = 50000
        num_steps = 256
        T = 2.0
        M = 50.0
        process = IntensityProcess(mu=mu, sigma=0.0, k=k, nu=nu)

        T_grid = np.linspace(1, int(T), int(M))
        lambda_grid = []
        for Tm in T_grid:
            mc = MonteCarloMethod(process, lambda_0, Tm, num_paths, num_steps)        
            paths = mc.simulate()
            lambda_m = torch.mean(paths[:, -1]).item()
            lambda_grid.append(lambda_m)


        time_points = torch.tensor(T_grid, dtype=torch.float32)
        dt = time_points[1:] - time_points[:-1]
        lambda_points = torch.tensor(lambda_grid, dtype=torch.float32)
        lambda_points = lambda_points[:-1]  # Truncate to match dt size

        survival_probs = torch.exp(-torch.sum(lambda_points * dt))
        print(survival_probs)

        # Calculate CVA
        def calculate_cva(LGD, S, K, T, r, survival_probs):
            payoffs = torch.maximum(S[:, -1] - K, torch.tensor(0.0))  # Max(S_T - K, 0)
            # Convert r and T to tensors before using them in torch.exp
            discount_factors = torch.exp(-torch.tensor(r, dtype=torch.float32) * T)
            cva = LGD * torch.mean((1 - survival_probs) * discount_factors * payoffs)
            return cva

        # Use inputs from self.option
        S0 = self.option.S0
        K = self.option.K
        #T = torch.tensor(self.option.T, dtype=torch.float32, requires_grad=True)
        T = self.option.T
        r = self.option.r
        sigma = self.option.sigma


        process = LogNormalProcess(r, sigma)
        mc = MonteCarloMethod(process, S0, T, num_paths, num_steps)        
        paths = mc.simulate()


        # Compute CVA using automatic differentiation
        cva = calculate_cva(LGD, paths, K, T, r, survival_probs)
        print(f"CIR Intensity CVA: {cva.item():.5f}")
        #print(f"Delta of CVA w.r.t lambda_0: {delta_cva:.5f}")




if __name__ == '__main__':
    unittest.main()
