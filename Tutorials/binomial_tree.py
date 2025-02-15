import numpy as np

def binomial_tree(S0, K, T, r, sigma, num_steps, is_call=True):
    """
    Price an option using the binomial tree method.

    Parameters:
    - S0: Initial stock price (float)
    - K: Strike price (float)
    - T: Time to maturity (float, in years)
    - r: Risk-free rate (float)
    - sigma: Volatility (float)
    - num_steps: Number of steps in the binomial tree (int)
    - is_call: Whether the option is a call (default True). False for a put option.

    Returns:
    - option_price: float
    """
    delta_t = T / num_steps
    u = np.exp(sigma * np.sqrt(delta_t))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(r * delta_t) - d) / (u - d)  # Risk-neutral probability
    q = 1 - p  # Complement probability

    # Initialize asset price tree
    S = np.zeros((num_steps + 1, num_steps + 1))
    S[0, 0] = S0
    for i in range(1, num_steps + 1):
        S[i, 0] = S[i - 1, 0] * u
        for j in range(1, i + 1):
            S[i, j] = S[i - 1, j - 1] * d

    # Initialize option value tree
    V = np.zeros((num_steps + 1, num_steps + 1))
    for j in range(num_steps + 1):
        if is_call:
            V[num_steps, j] = max(S[num_steps, j] - K, 0)  # Call option payoff
        else:
            V[num_steps, j] = max(K - S[num_steps, j], 0)  # Put option payoff

    # Backward induction
    for i in range(num_steps - 1, -1, -1):
        for j in range(i + 1):
            V[i, j] = np.exp(-r * delta_t) * (p * V[i + 1, j] + q * V[i + 1, j + 1])

    return V[0, 0]

# Example usage
if __name__ == "__main__":
    num_steps = 200
    S0 = 10
    K = 10
    sigma = 0.4
    r = 0.05
    T = 1

    call_price = binomial_tree(S0, K, T, r, sigma, num_steps, is_call=True)
    put_price = binomial_tree(S0, K, T, r, sigma, num_steps, is_call=False)
    print(f"Call Option Price: {call_price:.6f}")
    print(f"Put Option Price: {put_price:.6f}")
