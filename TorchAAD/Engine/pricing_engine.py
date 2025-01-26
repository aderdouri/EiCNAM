class PricingEngine:
    """Engine to manage pricing instruments."""
    def __init__(self, model):
        self.model = model

    def price_instrument(self, instrument, pricing_method):
        return instrument.price(pricing_method)
