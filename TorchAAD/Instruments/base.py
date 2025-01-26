class FinancialInstrument:
    """Base class for financial instruments."""
    def __init__(self, name):
        self.name = name

    def price(self, pricing_method):
        """Price the instrument using a given pricing method."""
        raise NotImplementedError("Price method must be implemented in subclasses.")
