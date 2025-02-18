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

class TestBinomialTreeMethod(unittest.TestCase):
    def test_price_bermudan_option_01(self):
        instrument = MockInstrument(
            name="Bermudan Option",
            S0=100.0,
            strike=95.0,
            maturity=180 / 365,
            rate=0.05,
            volatility=0.25
        )
        num_steps = 12
        exercise_times = [3/12, 6/12, 9/12, 12/12, 15/12, 18/12]
        method = BinomialTreeMethod(num_steps, exercise_times)
        price = method.price(instrument)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Bermudan Put Option Price: {price.item()}")

    def test_price_bermudan_option_strike_0_9(self):
        instrument = MockInstrument(
            name="Bermudan Option",
            S0=1.0,
            strike=0.9,
            maturity=3.0,
            rate=0.15,
            volatility=0.2
        )
        num_steps = 12
        exercise_times = [3/12, 6/12, 9/12, 12/12, 15/12, 18/12, 21/12, 24/12, 27/12, 30/12, 33/12, 36/12]
        method = BinomialTreeMethod(num_steps, exercise_times)
        price = method.price(instrument)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Bermudan Option Price with Strike 0.9: {price.item()}")

    def test_price_bermudan_option_strike_1_0(self):
        instrument = MockInstrument(
            name="Bermudan Option",
            S0=1.0,
            strike=1.0,
            maturity=3,
            rate=0.15,
            volatility=0.2
        )
        num_steps = 12
        exercise_times = [3/12, 6/12, 9/12, 12/12, 15/12, 18/12, 21/12, 24/12, 27/12, 30/12, 33/12, 36/12]
        method = BinomialTreeMethod(num_steps, exercise_times)
        price = method.price(instrument)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Bermudan Option Price with Strike 1.0: {price.item()}")

    def test_price_bermudan_option_strike_1_1(self):
        instrument = MockInstrument(
            name="Bermudan Option",
            S0=1.0,
            strike=1.1,
            maturity=3,
            rate=0.15,
            volatility=0.2
        )
        num_steps = 12
        exercise_times = [3/12, 6/12, 9/12, 12/12, 15/12, 18/12, 21/12, 24/12, 27/12, 30/12, 33/12, 36/12]
        method = BinomialTreeMethod(num_steps, exercise_times)
        price = method.price(instrument)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Bermudan Option Price with Strike 1.1: {price.item()}")

if __name__ == "__main__":
    unittest.main()