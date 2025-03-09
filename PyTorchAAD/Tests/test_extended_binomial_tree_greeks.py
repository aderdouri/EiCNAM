import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Methods.extended_binomial_tree import ExtendedBinomialTreeMethod

class TestExtendedBinomialTreeMethod(unittest.TestCase):
    def test_greeks_calculation(self):
        # Parameters
        S0 = 1.0   # Spot price of the asset
        K = 0.9    # Strike price
        r = 0.15    # Risk-free rate
        sigma = 0.20 # Volatility of the asset
        T = 3.0      # Time to expiration (3 years)
        num_steps = 50  # Number of steps in the binomial tree
        exercise_dates = [12, 24, 36, 48]  # Bermudan exercise points (every 3 months)

        # Initialize the extended binomial tree
        tree = ExtendedBinomialTreeMethod(num_steps)

        # Calculate Greeks
        greeks = tree.calculate_greeks(S0, K, T, r, sigma, exercise_dates, is_call=False)

        print(f"Delta: {greeks['Delta']}")
        print(f"Vega: {greeks['Vega']}")
        print(f"Rho: {greeks['Rho']}")
        print(f"Theta: {greeks['Theta']}")

        # Assert the Greeks are not None
        self.assertIsNotNone(greeks['Delta'])
        self.assertIsNotNone(greeks['Vega'])
        self.assertIsNotNone(greeks['Rho'])
        self.assertIsNotNone(greeks['Theta'])

if __name__ == "__main__":
    unittest.main()