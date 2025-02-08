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

    def calculate_cva(self, LGD, S, K, T, r, lambda_t, mu, k, nu, num_paths, num_steps):
        process = IntensityProcess(mu=mu, sigma=0.0, k=k, nu=nu)
        mc = MonteCarloMethod(process, lambda_t, T, num_paths, num_steps)        
        lambdas_paths = mc.simulate()

        dt_step = T / num_steps
        integrated_intensity = torch.sum(lambdas_paths.T[:, :-1] * dt_step, dim=1)  
        
        survival_probs = torch.exp(-integrated_intensity).mean()
        #print(f"Average survival probability: {survival_probs.item():.4f}")

        payoffs = torch.maximum(S[:, -1] - K, torch.tensor(0.0))
        discount_factors = torch.exp(-torch.tensor(r, dtype=torch.float32) * T)
        option_price = torch.mean(discount_factors * payoffs)
        #print(f"option_price: {option_price.item()}")
        cva = LGD * torch.mean((1 - survival_probs) * discount_factors * payoffs)
        return cva

    def test_cir_intensity_cva(self):
        lambda_0 = 1.0
        k = 0.5
        mu = 1.0
        nu = 0.25
        LGD = 0.6
        num_paths = 5000
        num_steps = 50
        T = 2.0

        lambda_t = torch.tensor(lambda_0, requires_grad=True)

        S0 = self.option.S0
        K = self.option.K
        T = self.option.T
        r = self.option.r
        sigma = self.option.sigma

        process = LogNormalProcess(r, sigma)
        mc = MonteCarloMethod(process, S0, T, num_paths, num_steps)        
        paths = mc.simulate()

        cva = self.calculate_cva(LGD, paths, K, T, r, lambda_t, mu, k, nu, num_paths, num_steps)
        print(f"CIR Intensity CVA: {cva.item():.5f}")

        cva.backward()
        delta_cva = lambda_t.grad.item()
        print(f"Delta of CVA w.r.t lambda_0: {delta_cva:.5f}")

        self.assertTrue(cva.item() > 0)
        self.assertTrue(delta_cva != 0)

    def test_cir_intensity_cva_different_S0(self):
        test_values = [110.0, 90.0]
        for S0 in test_values:
            self.option.S0 = S0
            lambda_0 = 1.0
            k = 0.5
            mu = 1.0
            nu = 0.25
            LGD = 0.6
            num_paths = 5000
            num_steps = 50
            T = 2.0

            lambda_t = torch.tensor(lambda_0, requires_grad=True)

            K = self.option.K
            T = self.option.T
            r = self.option.r
            sigma = self.option.sigma

            process = LogNormalProcess(r, sigma)
            mc = MonteCarloMethod(process, S0, T, num_paths, num_steps)        
            paths = mc.simulate()

            cva = self.calculate_cva(LGD, paths, K, T, r, lambda_t, mu, k, nu, num_paths, num_steps)
            print(f"CIR Intensity CVA for S0={S0}: {cva.item():.5f}")

            cva.backward()
            delta_cva = lambda_t.grad.item()
            print(f"Delta of CVA w.r.t lambda_0 for S0={S0}: {delta_cva:.5f}")

            self.assertTrue(cva.item() > 0)
            self.assertTrue(delta_cva != 0)

if __name__ == '__main__':
    unittest.main()
