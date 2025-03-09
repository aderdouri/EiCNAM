import unittest
import torch
from Methods.binomial_tree import BinomialTreeMethod

class MockInstrument:
    def __init__(self, name, S0, strike, maturity, rate, volatility):
        self.name = name
        self.S0 = S0
        self.strike = strike
        self.maturity = maturity
        self.rate = rate
        self.volatility = volatility

class TestBinomialTreeMethodCVA(unittest.TestCase):
    def test_calculate_cva(self):
        instrument = MockInstrument(
            name="Option",
            S0=1.0,
            strike=0.9,
            maturity=3,
            rate=0.15,
            volatility=0.2
        )
        num_steps = 12
        default_prob = 0.02  # Example default probability
        recovery_rate = 0.4  # Example recovery rate
        method = BinomialTreeMethod(num_steps)
        cva = method.calculate_cva(instrument, default_prob, recovery_rate)
        
        # Assert the CVA is a tensor and has a valid value
        self.assertIsInstance(cva, torch.Tensor)
        self.assertGreaterEqual(cva.item(), 0)
        print(f"CVA: {cva.item()}")

if __name__ == '__main__':
    unittest.main()
