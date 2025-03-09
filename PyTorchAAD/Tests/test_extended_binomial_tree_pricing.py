import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Methods.extended_binomial_tree import ExtendedBinomialTreeMethod

class TestExtendedBinomialTreeMethod(unittest.TestCase):

    def test_bermudan_put_option(self):
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

        # Price the option and get gradients
        option_price = tree.price(S0, K, T, r, sigma, exercise_dates, is_call=False)
        
        # Convert tensor to float
        option_price = option_price.item()

        print(f"Bermudan put option price: {option_price:.7f}")
        
        # Assert the option price is as expected (replace 0.0 with the expected value)
        #self.assertAlmostEqual(option_price, 0.0, places=6)
        # Assert the gradient is as expected (replace 0.0 with the expected value)
        #self.assertAlmostEqual(gradients, 0.0, places=6)
if __name__ == "__main__":
    unittest.main()