import unittest
from TorchAAD.Engine.stochastic_process import NormalProcess, LogNormalProcess
from TorchAAD.Engine.simulator import simulate_process

class TestSimulations(unittest.TestCase):

    def test_normal_process(self):
        process = NormalProcess(mu=0.1, sigma=0.2)
        paths = simulate_process(process, S0=100, T=1.0, steps=10, n_paths=5)
        self.assertEqual(paths.shape, (10, 5))

    def test_lognormal_process(self):
        process = LogNormalProcess(mu=0.1, sigma=0.2)
        paths = simulate_process(process, S0=100, T=1.0, steps=10, n_paths=5)
        self.assertEqual(paths.shape, (10, 5))

if __name__ == '__main__':
    unittest.main()
