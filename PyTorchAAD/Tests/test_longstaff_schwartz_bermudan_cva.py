import unittest
import torch
import json
from Methods.longstaff_schwartz import LongstaffSchwartzMethod

class MockInstrument:
    def __init__(self, name, S0, strike, maturity, rate, volatility):
        self.name = name
        self.S0 = S0
        self.strike = strike
        self.maturity = maturity
        self.rate = rate
        self.volatility = volatility

class TestLongstaffSchwartzMethod(unittest.TestCase):
    def test_greeks_bermudan_option_01(self):
        instrument = MockInstrument(
            name="Bermudan Option",
            S0=1.0,
            strike=0.9,
            maturity=3.0,
            rate=0.15,
            volatility=0.20
        )
        method = LongstaffSchwartzMethod(num_paths=1000, num_steps=50)
        price = method.price(instrument.S0, instrument.strike, 
                             instrument.volatility, instrument.maturity, instrument.rate)
        greeks = method.calculate_greeks(instrument.S0, instrument.strike, 
                                         instrument.volatility, instrument.maturity, instrument.rate)
        
        self.assertIsInstance(greeks, dict)
        print(f"Price for Bermudan Option: {price.item()}")
        print(f"Greeks for Bermudan Option: {json.dumps(greeks, indent=4)}")

if __name__ == '__main__':
    unittest.main()
