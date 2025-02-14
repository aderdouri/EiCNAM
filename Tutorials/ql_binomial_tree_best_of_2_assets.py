import QuantLib as ql

def price_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='put', steps=500):
    """
    Pricing a Best-of-Two Assets Option using a Binomial Tree (CRR Model).
    """
    # 1️⃣ Set evaluation date
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today

    # 2️⃣ Define the effective asset process
    if option_type == 'put':
        Seff = min(S1, S2)
    elif option_type == 'call':
        Seff = max(S1, S2)
    else:
        raise ValueError("option_type must be 'put' or 'call'")
    
    sigma_eff = (sigma1 + sigma2) / 2  # Approximate average volatility

    # 3️⃣ Define Black-Scholes process for the synthetic asset
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(Seff))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(r)), ql.Actual360()))
    dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.Actual360()))
    vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(sigma_eff)), ql.Actual360()))

    process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, vol_handle)

    # 4️⃣ Set up binomial pricing engine
    engine = ql.BinomialVanillaEngine(process, "CoxRossRubinstein", steps)

    # 5️⃣ Define the Best-of-Two Assets Option
    if option_type == 'put':
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    elif option_type == 'call':
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    
    exercise = ql.EuropeanExercise(today + ql.Period(int(T * 365), ql.Days))
    option = ql.VanillaOption(payoff, exercise)

    # 6️⃣ Compute the price
    option.setPricingEngine(engine)
    return option.NPV()

# S10 = 90, S20 = 100, K = 100, r= 0.04, T = 1, σ= 0.4, M = 50

# 🔹 Example Pricing for a Best-of-Two Assets Option
S1 = 90.0    # Spot price of asset 1
S2 = 100.0    # Spot price of asset 2
K = 100.0     # Strike price
r = 0.04      # Risk-free rate
sigma1 = 0.4  # Volatility of asset 1
sigma2 = 0.4 # Volatility of asset 2
rho = 0.0     # Correlation between assets
T = 1.0       # Time to expiration (1 year)
steps = 500   # Binomial tree steps

# Compute the option price
price = price_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='put')
print(f"Best-of-Two Assets Put Option Price: {price:.6f}")

price = price_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='call')
print(f"Best-of-Two Assets Call Option Price: {price:.6f}")
