class PricingEngine:
    """Engine to manage pricing instruments."""
    def __init__(self, model):
        self.model = model

    def price_instrument(self, instrument, pricing_method):
        # Consider that the option can be exercised every three months (M = 12)
        M = 12
        return instrument.price(pricing_method, M)
