import unittest
from Instruments.european_option import EuropeanOption
from Methods.monte_carlo import MonteCarloMethod

class MockInstrument:
    def __init__(self, name, S0, strike, maturity, rate, volatility):
        self.name = name
        self.S0 = S0
        self.strike = strike
        self.maturity = maturity
        self.rate = rate
        self.volatility = volatility
class TestEuropeanOption(unittest.TestCase):
    def setUp(self):
        self.option = EuropeanOption(S0=100.0, K=90.0, T=2.0, r=0.01, sigma=0.25)

    def test_initialization(self):
        self.assertEqual(self.option.S0, 100)
        self.assertEqual(self.option.K, 90)
        self.assertEqual(self.option.T, 2.0)
        self.assertEqual(self.option.r, 0.01)
        self.assertEqual(self.option.sigma, 0.25)


    def test_price_bermudan_option_01(self):
        instrument = MockInstrument(
            name="European Option",
            S0=100.0,
            strike=90.0,
            maturity=2.0,
            rate=0.01,
            volatility=0.25
        )

        method = MonteCarloMethod(num_paths=1000, num_steps=50)
        price = method.price(instrument)
        print(f"European Option Price: {price}")

if __name__ == '__main__':
    unittest.main()
