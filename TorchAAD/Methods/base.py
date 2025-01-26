class PricingMethod:
    """Base class for pricing methods."""
    def price(self, instrument):
        """Price the given instrument."""
        raise NotImplementedError("Price method must be implemented in subclasses.")
