import unittest
import torch
import numpy as np
from Instruments.european_option import EuropeanOption
from Engine.stochastic_process import LogNormalProcess, IntensityProcess, CIRPlusPlusProcess
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

    def calculate_cva(self, LGD, S, K, T, r, lambda_t, mu, k, nu, theta, phi, 
                      num_paths, num_steps):
        
        process = CIRPlusPlusProcess(mu=mu, sigma=0.0, k=k, theta=theta, nu=nu, phi=phi)
        mc = MonteCarloMethod(process, lambda_t, T, num_paths, num_steps)        
        lambdas_paths = mc.simulate()

        time_grid = torch.linspace(0, T, num_steps)
        lambdas_paths = lambdas_paths + phi(time_grid)  # Add deterministic shift

        dt_step = T / num_steps
        integrated_hazard = torch.cumsum(lambdas_paths[:, :-1] * dt_step, dim=1)
        print(f"Average integrated intensity: {torch.mean(integrated_hazard).item():.4f}")        
        
        survival_probs = torch.exp(-integrated_hazard[:, -1])
        print(f"Average survival probability: {torch.mean(survival_probs).item():.4f}")

        # Compute payoff and CVA
        payoffs = torch.relu(S[:, -1] - K)
        discount_factors = torch.exp(-torch.tensor(r, dtype=torch.float32) * T)
        option_price = torch.mean(discount_factors * payoffs)
        print(f"Option price: {option_price.item():.5f}")
        cva = LGD * torch.mean((1 - survival_probs) * discount_factors * payoffs)
        return cva

    def test_cir_plus_plus_cva(self):
        print('------------------------------------')
        lambda_0 = 1.0
        k = 0.5
        mu = 1.0
        nu = 0.25
        theta = 0.1
        phi = lambda t: 0.005 * torch.exp(-0.1 * t)
        LGD = 0.6
        num_paths = 100000
        num_steps = 365
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

        cva = self.calculate_cva(LGD, paths, K, T, r, lambda_t, mu, k, nu, theta, phi, 
                                 num_paths, num_steps)
                
        print(f"CIR++ Intensity CVA: {cva.item():.5f}")

        cva.backward()
        delta_cva = lambda_t.grad.item()
        print(f"Delta of CVA w.r.t lambda_0: {delta_cva:.5f}")

        self.assertTrue(cva.item() > 0)
        self.assertTrue(delta_cva != 0)

    def test_cir_plus_plus_cva_different_S0(self):
        test_values = [110.0, 90.0]
        for S0 in test_values:
            print('------------------------------------')
            self.option.S0 = S0
            lambda_0 = 1.0
            k = 0.5
            mu = 1.0
            nu = 0.25
            theta = 0.1
            phi = lambda t: 0.005 * torch.exp(-0.1 * t)
            LGD = 0.6
            num_paths = 50000
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

            cva = self.calculate_cva(LGD, paths, K, T, r, lambda_t, mu, k, nu, theta, phi, num_paths, num_steps)
            print(f"CIR++ Intensity CVA for S0={S0}: {cva.item():.5f}")

            cva.backward()
            delta_cva = lambda_t.grad.item()
            print(f"Delta of CVA w.r.t lambda_0 for S0={S0}: {delta_cva:.5f}")

            self.assertTrue(cva.item() > 0)
            self.assertTrue(delta_cva != 0)

    def test_cir_plus_plus_cva_different_K(self):
        print('------------------------------------')
        lambda_0 = 1.0
        k = 0.5
        mu = 1.0
        nu = 0.25
        theta = 0.1
        phi = lambda t: 0.005 * torch.exp(-0.1 * t)
        LGD = 0.6
        num_paths = 50000
        num_steps = 50
        T = 2.0

        lambda_t = torch.tensor(lambda_0, requires_grad=True)

        S0 = self.option.S0
        K = 100.0
        T = self.option.T
        r = self.option.r
        sigma = self.option.sigma

        process = LogNormalProcess(r, sigma)
        mc = MonteCarloMethod(process, S0, T, num_paths, num_steps)        
        paths = mc.simulate()

        cva = self.calculate_cva(LGD, paths, K, T, r, lambda_t, mu, k, nu, theta, phi, num_paths, num_steps)
        print(f"CIR++ Intensity CVA for K={K}: {cva.item():.5f}")

        cva.backward()
        delta_cva = lambda_t.grad.item()
        print(f"Delta of CVA w.r.t lambda_0 for K={K}: {delta_cva:.5f}")

        self.assertTrue(cva.item() > 0)
        self.assertTrue(delta_cva != 0)

    def test_cir_plus_plus_cva_different_T(self):
        print('------------------------------------')
        lambda_0 = 1.0
        k = 0.5
        mu = 1.0
        nu = 0.25
        theta = 0.1
        phi = lambda t: 0.005 * torch.exp(-0.1 * t)
        LGD = 0.6
        num_paths = 50000
        num_steps = 50
        T = 1.0

        lambda_t = torch.tensor(lambda_0, requires_grad=True)

        S0 = self.option.S0
        K = self.option.K
        T = self.option.T
        r = self.option.r
        sigma = self.option.sigma

        process = LogNormalProcess(r, sigma)
        mc = MonteCarloMethod(process, S0, T, num_paths, num_steps)        
        paths = mc.simulate()

        cva = self.calculate_cva(LGD, paths, K, T, r, lambda_t, mu, k, nu, theta, phi, num_paths, num_steps)
        print(f"CIR++ Intensity CVA for T={T}: {cva.item():.5f}")

        cva.backward()
        delta_cva = lambda_t.grad.item()
        print(f"Delta of CVA w.r.t lambda_0 for T={T}: {delta_cva:.5f}")

        self.assertTrue(cva.item() > 0)
        self.assertTrue(delta_cva != 0)

    def test_cir_plus_plus_cva_different_sigma(self):
        print('------------------------------------')
        lambda_0 = 1.0
        k = 0.5
        mu = 1.0
        nu = 0.25
        theta = 0.1
        phi = lambda t: 0.005 * torch.exp(-0.1 * t)
        LGD = 0.6
        num_paths = 50000
        num_steps = 50
        T = 1.0

        lambda_t = torch.tensor(lambda_0, requires_grad=True)

        S0 = self.option.S0
        K = self.option.K
        T = self.option.T
        r = self.option.r
        sigma = 0.4

        process = LogNormalProcess(r, sigma)
        mc = MonteCarloMethod(process, S0, T, num_paths, num_steps)        
        paths = mc.simulate()

        cva = self.calculate_cva(LGD, paths, K, T, r, lambda_t, mu, k, nu, theta, phi, num_paths, num_steps)
        print(f"CIR++ Intensity CVA for sigma={sigma}: {cva.item():.5f}")

        cva.backward()
        delta_cva = lambda_t.grad.item()
        print(f"Delta of CVA w.r.t lambda_0 for sigma={sigma}: {delta_cva:.5f}")

        self.assertTrue(cva.item() > 0)
        self.assertTrue(delta_cva != 0)


if __name__ == '__main__':
    unittest.main()
