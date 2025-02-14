import unittest
import torch
from Methods.longstaff_schwartz_best_of_two_assets import LongstaffSchwartzMethodBestOf2Assets

class MockInstrument:
    def __init__(self, name, S0_1, S0_2, strike, maturity, rate, volatility1, volatility2):
        self.name = name
        self.S0_1 = S0_1
        self.S0_2 = S0_2
        self.strike = strike
        self.maturity = maturity
        self.rate = rate
        self.volatility1 = volatility1
        self.volatility2 = volatility2

class TestLongstaffSchwartzMethodBestOf2Assets(unittest.TestCase):
    def test_price_best_of_two_put_option_01(self):

        instrument = MockInstrument(
            name="Best of Two Put Option",
            S0_1=90.0,
            S0_2=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.04,
            volatility1=0.40,
            volatility2=0.40
        )
        method = LongstaffSchwartzMethodBestOf2Assets(num_paths=50000, num_steps=365)
        price = method.price(instrument.S0_1, instrument.S0_2, instrument.strike, 
                                             instrument.volatility1, instrument.volatility2, 
                                             instrument.maturity, instrument.rate, M=100)
        
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Longstaff-Schwartz price for Best of Two Put Option: {price.item()}")


    def test_price_best_of_two_put_option_02(self):

        instrument = MockInstrument(
            name="Best of Two Put Option",
            S0_1=1.0,
            S0_2=1.0,
            strike=0.9,
            maturity=3.0,
            rate=0.15,
            volatility1=0.20,
            volatility2=0.20
        )
        method = LongstaffSchwartzMethodBestOf2Assets(num_paths=50000, num_steps=360)
        price = method.price(instrument.S0_1, instrument.S0_2, instrument.strike, 
                                             instrument.volatility1, instrument.volatility2, 
                                             instrument.maturity, instrument.rate, M=30)
        
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Longstaff-Schwartz price for Best of Two Put Option: {price.item()}")


    def test_price_best_of_two_call_option_01(self):

        instrument = MockInstrument(
            name="Best of Two Put Option",
            S0_1=1.0,
            S0_2=1.0,
            strike=0.9,
            maturity=3.0,
            rate=0.15,
            volatility1=0.20,
            volatility2=0.20
        )
        method = LongstaffSchwartzMethodBestOf2Assets(num_paths=50000, num_steps=360)
        price = method.price(instrument.S0_1, instrument.S0_2, instrument.strike, 
                                             instrument.volatility1, instrument.volatility2, 
                                             instrument.maturity, instrument.rate, M=30, option_type='call')
        
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Longstaff-Schwartz price for Best of Two Call Option: {price.item()}")


    def test_price_best_of_two_call_option_02(self):

        instrument = MockInstrument(
            name="Best of Two Call Option",
            S0_1=1.0,
            S0_2=1.0,
            strike=0.9,
            maturity=3.0,
            rate=0.15,
            volatility1=0.20,
            volatility2=0.20
        )
        method = LongstaffSchwartzMethodBestOf2Assets(num_paths=50000, num_steps=360)
        price = method.price(instrument.S0_1, instrument.S0_2, instrument.strike, 
                                             instrument.volatility1, instrument.volatility2, 
                                             instrument.maturity, instrument.rate, M=30, option_type='call')
        
        self.assertIsInstance(price, torch.Tensor)
        self.assertGreater(price.item(), 0)
        print(f"Longstaff-Schwartz price for Best of Two Call Option: {price.item()}")

if __name__ == '__main__':
    unittest.main()
