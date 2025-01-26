import unittest
import torch
from TorchAAD.Methods.binomial_tree import BinomialTreeMethod
unittest.TestLoader.testMethodPrefix = "_not_a_test"

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
            S0=1.0,
            strike=0.9,
            maturity=3,
            rate=0.15,
            volatility=0.2
        )
        num_steps = 12
        exercise_dates = [3, 6, 9, 12]  # Exercise every 3 months
        method = BinomialTreeMethod(num_steps)
        price = method.price_bermudan_option(instrument, exercise_dates)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Bermudan Put Option Price: {price.item()}")

    def test_price_bermudan_put_option_02(self):
        instrument = MockInstrument(
            name="Bermudan Put Option",
            S0=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.4
        )
        num_steps = 12
        exercise_dates = [3, 6, 9, 12]  # Exercise every 3 months
        method = BinomialTreeMethod(num_steps)
        price = method.price_bermudan_option(instrument, exercise_dates)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Bermudan Put Option Price: {price.item()}")

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
        method = BinomialTreeMethod(num_steps)
        price = method.price_bermudan_option(instrument, exercise_dates)
        
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
        exercise_dates = [3, 6, 9, 12]  # Exercise every 3 months
        method = BinomialTreeMethod(num_steps)
        price = method.price_bermudan_option(instrument, exercise_dates)
        
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
        exercise_dates = [3, 6, 9, 12]  # Exercise every 3 months
        method = BinomialTreeMethod(num_steps)
        price = method.price_bermudan_option(instrument, exercise_dates)
        
        # Assert the price is a tensor and has a valid value
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Bermudan Option Price with Strike 1.1: {price.item()}")

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestBinomialTreeMethod('test_price_bermudan_option_01'))
    suite.addTest(TestBinomialTreeMethod('test_price_bermudan_put_option_02'))
    suite.addTest(TestBinomialTreeMethod('test_price_bermudan_option_strike_0_9'))
    suite.addTest(TestBinomialTreeMethod('test_price_bermudan_option_strike_1_0'))
    suite.addTest(TestBinomialTreeMethod('test_price_bermudan_option_strike_1_1'))

    runner = unittest.TextTestRunner()
    print("Running tests in specified order:")
    runner.run(suite)
