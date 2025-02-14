import QuantLib as ql

def generate_exercise_dates(start_date, maturity_years, frequency, unit):
    if unit == 'months':
        num_exercise_dates = int(maturity_years * 12 / frequency)
        return [start_date + ql.Period(i * frequency, ql.Months) for i in range(1, num_exercise_dates + 1)]
    elif unit == 'weeks':
        num_exercise_dates = int(maturity_years * 52 / frequency)
        return [start_date + ql.Period(i * frequency, ql.Weeks) for i in range(1, num_exercise_dates + 1)]
    elif unit == 'days':
        num_exercise_dates = int(maturity_years * 365 / frequency)
        return [start_date + ql.Period(i * frequency, ql.Days) for i in range(1, num_exercise_dates + 1)]
    else:
        raise ValueError("unit must be 'months', 'weeks', or 'days'")

def price_bermudan_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='put', steps=500, frequency=3, unit='months'):
    """
    Pricing a Bermudan Best-of-Two Assets Option using a Binomial Tree (CRR Model).
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

    # 5️⃣ Generate Bermudan exercise dates
    exercise_dates_list = generate_exercise_dates(today, T, frequency, unit)
    print(len(exercise_dates_list))

    # 6️⃣ Define the Bermudan Best-of-Two Assets Option
    if option_type == 'put':
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    elif option_type == 'call':
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    
    exercise = ql.BermudanExercise(exercise_dates_list)
    option = ql.VanillaOption(payoff, exercise)

    # 7️⃣ Compute the price
    option.setPricingEngine(engine)
    return option.NPV()

# 🔹 Example Pricing for a Bermudan Best-of-Two Assets Option

# S10 = 90, S20 = 100, K = 100, r= 0.04, T = 1, σ= 0.4, M = 50

S1 = 90.0    # Spot price of asset 1
S2 = 100.0    # Spot price of asset 2
K = 100.0     # Strike price
r = 0.04      # Risk-free rate
sigma1 = 0.40  # Volatility of asset 1
sigma2 = 0.40 # Volatility of asset 2
rho = 0.0     # Correlation between assets
T = 3.0       # Time to expiration (1 year)
steps = 500   # Binomial tree steps
frequency = 1  # Exercise frequency
unit = 'weeks'  # Exercise frequency unit

# Compute the option price
price = price_bermudan_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='put', steps=steps, 
                                                 frequency=frequency, unit=unit)
print(f"Bermudan Best-of-Two Assets Put Option Price: {price:.6f}")


price = price_bermudan_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='call', steps=steps, 
                                                 frequency=frequency, unit=unit)
print(f"Bermudan Best-of-Two Assets Call Option Price: {price:.6f}")

print("-----------------------------------\n")


# S10 = 1.0, S20 = 1.0, K = 0.9, r = 0.15, T = 3.0, σ= 0.2, M = 12 and 13 basis functions.

S1 = 1.0    # Spot price of asset 1
S2 = 1.0    # Spot price of asset 2
K = 0.9     # Strike price
r = 0.15      # Risk-free rate
sigma1 = 0.20  # Volatility of asset 1
sigma2 = 0.20 # Volatility of asset 2
rho = 0.0     # Correlation between assets
T = 3.0       # Time to expiration (3 years)
steps = 500   # Binomial tree steps
frequency = 3  # Exercise frequency
unit = 'months'  # Exercise frequency unit

# Compute the option price
price = price_bermudan_best_of_two_assets_option(S1, S2, K, r, sigma1, sigma2, rho, T, option_type='put', 
                                                 steps=steps, frequency=frequency, unit=unit)
print(f"Bermudan Best-of-Two Assets Put Option Price: {price:.6f}")
