import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Methods.monte_carlo_pricing import MonteCarloPricing

class TestMonteCarloPricing(unittest.TestCase):
    def test_best_of_two_assets_bermudan_option(self):
        # Parameters
        S0_1 = 90.0  # Initial price of asset 1
        S0_2 = 100.0  # Initial price of asset 2
        K = 100.0     # Strike price
        T = 1.0       # Time to maturity (1 year)
        r = 0.04      # Risk-free rate
        sigma_1 = 0.4 # Volatility of asset 1
        sigma_2 = 0.4 # Volatility of asset 2
        rho = 0.0     # Correlation between the two assets
        num_paths = 500000  # Number of Monte Carlo paths
        num_steps = 1000     # Number of time steps
        exercise_dates = list(range(1, 51))  # Bermudan exercise points (50 dates)

        # Initialize Monte Carlo pricing
        mc_pricing = MonteCarloPricing(num_paths, num_steps)

        # Price the call option
        call_option_price = mc_pricing.price_best_of_two_assets_bermudan_option(S0_1, S0_2, K, T, r, sigma_1, sigma_2, rho, exercise_dates, is_call=True)
        print(f"Best of Two Assets Bermudan Call Option Price: {call_option_price:.6f}")

        # Price the put option
        put_option_price = mc_pricing.price_best_of_two_assets_bermudan_option(S0_1, S0_2, K, T, r, sigma_1, sigma_2, rho, exercise_dates, is_call=False)
        print(f"Best of Two Assets Bermudan Put Option Price: {put_option_price:.6f}")

        # Assert the option prices are as expected (replace 0.0 with the expected values)
        #self.assertAlmostEqual(call_option_price, 0.0, places=6)
        #self.assertAlmostEqual(put_option_price, 0.0, places=6)

    def test_best_of_two_assets_bermudan_option_new_params(self):
        # Parameters
        S0_1 = 1.0   # Initial price of asset 1
        S0_2 = 1.0   # Initial price of asset 2
        K = 0.9      # Strike price
        T = 3.0      # Time to maturity (3 years)
        r = 0.15     # Risk-free rate
        sigma_1 = 0.2 # Volatility of asset 1
        sigma_2 = 0.2 # Volatility of asset 2
        rho = 0.0    # Correlation between the two assets
        num_paths = 500000  # Number of Monte Carlo paths
        num_steps = 1000    # Number of time steps
        exercise_dates = list(range(1, 13))  # Bermudan exercise points (12 dates)

        # Initialize Monte Carlo pricing
        mc_pricing = MonteCarloPricing(num_paths, num_steps)

        # Price the call option
        call_option_price = mc_pricing.price_best_of_two_assets_bermudan_option(S0_1, S0_2, K, T, r, sigma_1, sigma_2, rho, exercise_dates, is_call=True)
        print(f"Best of Two Assets Bermudan Call Option Price (new params): {call_option_price:.6f}")

        # Price the put option
        put_option_price = mc_pricing.price_best_of_two_assets_bermudan_option(S0_1, S0_2, K, T, r, sigma_1, sigma_2, rho, exercise_dates, is_call=False)
        print(f"Best of Two Assets Bermudan Put Option Price (new params): {put_option_price:.6f}")

        # Assert the option prices are as expected (replace 0.0 with the expected values)
        #self.assertAlmostEqual(call_option_price, 0.0, places=6)
        #self.assertAlmostEqual(put_option_price, 0.0, places=6)

if __name__ == "__main__":
    unittest.main()
