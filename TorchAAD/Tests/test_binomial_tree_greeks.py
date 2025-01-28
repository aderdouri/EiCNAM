import unittest
import torch
import json
from TorchAAD.Methods.binomial_tree import BinomialTreeMethod

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
        exercise_dates = [3, 6, 9, 12]  # Exercise every 3 months
        method = BinomialTreeMethod(num_steps, exercise_dates)
        price = method.price(instrument)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.clone().detach().item(), 0)
        print(f"Bermudan Put Option Price: {price.clone().detach().item()}")

    def test_price_bermudan_option_strike_0_9(self):
        instrument = MockInstrument(
            name="Bermudan Option",
            S0=1.0,
            strike=0.9,
            maturity=3,
            rate=0.15,
            volatility=0.2
        )
        num_steps = 12
        exercise_dates = [3, 6, 9, 12]  # Exercise every 3 months
        method = BinomialTreeMethod(num_steps, exercise_dates)
        price = method.price(instrument)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.clone().detach().item(), 0)
        print(f"Bermudan Option Price with Strike 0.9: {price.clone().detach().item()}")

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
        exercise_dates = [3, 6, 9, 12]  # Exercise every 3 months
        method = BinomialTreeMethod(num_steps, exercise_dates)
        price = method.price(instrument)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.clone().detach().item(), 0)
        print(f"Bermudan Option Price with Strike 1.0: {price.clone().detach().item()}")

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
        exercise_dates = [3, 6, 9, 12]  # Exercise every 3 months
        method = BinomialTreeMethod(num_steps, exercise_dates)
        price = method.price(instrument)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.clone().detach().item(), 0)
        print(f"Bermudan Option Price with Strike 1.1: {price.clone().detach().item()}")

    def test_greeks_bermudan_option(self):
        instrument = MockInstrument(
            name="Bermudan Option",
            S0=100.0,
            strike=95.0,
            maturity=180 / 365,
            rate=0.05,
            volatility=0.25
        )
        num_steps = 12
        exercise_dates = [3, 6, 9, 12]  # Exercise every 3 months
        method = BinomialTreeMethod(num_steps, exercise_dates)
        greeks = method.calculate_greeks(instrument)
        
        # Beautified print statement for Greeks
        print(f"Greeks for Bermudan Option with Strike 1.1: {json.dumps(greeks, indent=4)}")

if __name__ == "__main__":
    unittest.main()