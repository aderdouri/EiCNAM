from .base import FinancialInstrument

class EuropeanOption(FinancialInstrument):
    def __init__(self, S0, K, T, r, sigma, option_type="call"):
        super().__init__("European Option")
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type

    def price(self, pricing_method):
        return pricing_method.price(self)
