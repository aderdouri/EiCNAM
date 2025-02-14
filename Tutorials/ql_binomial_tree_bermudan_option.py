import QuantLib as ql

def generate_exercise_dates(start_date, maturity_years, frequency_months):
    num_exercise_dates = int(maturity_years * 12 / frequency_months)
    return [start_date + ql.Period(i * frequency_months, ql.Months) for i in range(1, num_exercise_dates + 1)]

def price_bermudan_option(S0, K, r, sigma, T, steps, option_type='put', frequency_months=3):
    # 1️⃣ Set up the market parameters
    # ...existing code...

    # 2️⃣ Define QuantLib objects
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today

    # Generate exercise dates
    exercise_dates_list = generate_exercise_dates(today, T, frequency_months)
    # ...existing code...

    # Define the Black-Scholes process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(r)), ql.Actual360()))
    dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.Actual360()))  
    vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(sigma)), ql.Actual360()))

    # Fix: Ensure `dividend_handle` is properly initialized
    process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, vol_handle)

    # 3️⃣ Set up the binomial pricing engine
    engine = ql.BinomialVanillaEngine(process, "CoxRossRubinstein", steps)

    # 4️⃣ Define the Bermudan Option
    if option_type == 'put':
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    elif option_type == 'call':
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    else:
        raise ValueError("option_type must be 'put' or 'call'")
    
    exercise = ql.BermudanExercise(exercise_dates_list)
    option = ql.VanillaOption(payoff, exercise)

    # 5️⃣ Set the pricing engine and compute the price
    option.setPricingEngine(engine)
    price = option.NPV()

    return price

# Test for different values of K and option types
Ks = [0.9, 1.0, 1.1]
option_types = ['put']
for option_type in option_types:
    for K in Ks:
        price = price_bermudan_option(S0=1.0, K=K, r=0.15, sigma=0.2, T=3.0, steps=500, option_type=option_type)
        print(f"Bermudan {option_type.capitalize()} Option Price with Strike {K}: {price:.6f}")


#S10 = 90, S20 = 100, K = 100, r= 0.04, T = 1, σ= 0.4, M = 50


price = price_bermudan_option(S0=100.0, K=60, r=0.04, sigma=0.4, T=1.0, steps=500, option_type='put')
print(f"Bermudan {option_type.capitalize()} Option Price with Strike {K}: {price:.6f}")
