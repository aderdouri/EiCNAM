import unittest
from Instruments.european_option import EuropeanOption

class TestEuropeanOption(unittest.TestCase):
    def setUp(self):
        self.option = EuropeanOption(S0=100, K=110, T=1, r=0.05, sigma=0.2)

    def test_initialization(self):
        self.assertEqual(self.option.S0, 100)
        self.assertEqual(self.option.K, 110)
        self.assertEqual(self.option.T, 1)
        self.assertEqual(self.option.r, 0.05)
        self.assertEqual(self.option.sigma, 0.2)

    def test_price_method(self):
        with self.assertRaises(NotImplementedError):
            self.option.price("dummy_method")

if __name__ == '__main__':
    unittest.main()
