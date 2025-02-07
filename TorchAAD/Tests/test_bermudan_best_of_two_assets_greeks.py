import unittest
import torch
import json
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
    def test_greeks_best_of_two_put_option_01(self):

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
        method = LongstaffSchwartzMethodBestOf2Assets(num_paths=1000, num_steps=50)
        price = method.price(instrument.S0_1, instrument.S0_2, instrument.strike, 
                                             instrument.volatility1, instrument.volatility2, 
                                             instrument.maturity, instrument.rate)
        
        greeks = method.calculate_greeks(instrument.S0_1, instrument.S0_2, instrument.strike,
                                         instrument.volatility1, instrument.volatility2, 
                                         instrument.maturity, instrument.rate)
        
        self.assertIsInstance(greeks, dict)
        print(f"Price for Bermudan Option: {price.item()}")
        print(f"Greeks for Bermudan Option: {json.dumps(greeks, indent=4)}")


    def test_greeks_best_of_two_put_option_02(self):

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
        method = LongstaffSchwartzMethodBestOf2Assets(num_paths=1000, num_steps=12)
        price = method.price(instrument.S0_1, instrument.S0_2, instrument.strike, 
                                             instrument.volatility1, instrument.volatility2, 
                                             instrument.maturity, instrument.rate)

        greeks = method.calculate_greeks(instrument.S0_1, instrument.S0_2, instrument.strike,
                                         instrument.volatility1, instrument.volatility2, 
                                         instrument.maturity, instrument.rate)
        
        self.assertIsInstance(greeks, dict)
        print(f"Price for Bermudan Option: {price.item()}")
        print(f"Greeks for Bermudan Option: {json.dumps(greeks, indent=4)}")

if __name__ == '__main__':
    unittest.main()
