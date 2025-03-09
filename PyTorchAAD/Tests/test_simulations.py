import unittest
from Engine.stochastic_process import NormalProcess, LogNormalProcess, IntensityProcess
from Engine.simulator import simulate_process

import torch

class TestSimulations(unittest.TestCase):

    def test_normal_process(self):
        pass

    def test_lognormal_process(self):
        pass

    def test_intensity_process(self):
        process = IntensityProcess(mu=1.0, sigma=0.0, k=0.5, nu=0.2)
        # Initialize process
        S0 = torch.tensor(1.0)  # Initial value
        dt = 1 / 252            # Daily time step
        dW = torch.randn(1)     # Random normal increment

        # Evolve process
        S_next = process.evolve(S0, dt, dW)
        print(S_next)

        
                

if __name__ == '__main__':
    unittest.main()
