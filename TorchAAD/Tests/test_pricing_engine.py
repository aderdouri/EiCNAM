import unittest
from Engine.pricing_engine import PricingEngine
from Models.black_scholes import BlackScholesModel
from Instruments.european_option import EuropeanOption
from Methods.monte_carlo import MonteCarloMethod

class TestPricingEngine(unittest.TestCase):
    def setUp(self):
        self.model = BlackScholesModel(r=0.05, sigma=0.2)
        self.pricing_engine = PricingEngine(model=self.model)
        self.option = EuropeanOption(S0=100, K=110, T=1, r=0.05, sigma=0.2)
        self.method = MonteCarloMethod(num_paths=10000, num_steps=100)

    def test_price_instrument(self):
        price = self.pricing_engine.price_instrument(self.option, self.method)
        self.assertIsInstance(price, float)

if __name__ == '__main__':
    unittest.main()
