import unittest
import sys
import os

from Instruments.bermudan_option import BermudanOption

class TestBermudanOption(unittest.TestCase):
    def setUp(self):
        self.option = BermudanOption(S0=100, K=110, T=1, r=0.05, sigma=0.2, exercise_dates=[0.25, 0.5, 0.75, 1.0])

    def test_initialization(self):
        self.assertEqual(self.option.S0, 100)
        self.assertEqual(self.option.K, 110)
        self.assertEqual(self.option.T, 1)
        self.assertEqual(self.option.r, 0.05)
        self.assertEqual(self.option.sigma, 0.2)
        self.assertEqual(self.option.exercise_dates, [0.25, 0.5, 0.75, 1.0])

    def test_price_method(self):
        with self.assertRaises(NotImplementedError):
            self.option.price("dummy_method")

if __name__ == '__main__':
    unittest.main()
