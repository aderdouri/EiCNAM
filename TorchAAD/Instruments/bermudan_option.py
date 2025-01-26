from .base import FinancialInstrument

class BermudanOption(FinancialInstrument):
    def __init__(self, S0, K, T, r, sigma, exercise_dates):
        super().__init__("Bermudan Option")
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.exercise_dates = exercise_dates

    def price(self, pricing_method):
        return pricing_method.price(self)
