import numpy as np

# Parameters
eps = 1e-20  # For CVT
r = 0.05
sig = 0.5  # Replace with sig = 0.5 + 1j*eps for CVT-based Vega
T = 1
S0 = 100 + 1j * eps  # Inserting complex part for CVT
K = 110

# Monte Carlo simulation parameters
M = int(1e5)  # Total number of Monte Carlo paths
M2 = int(1e4)  # Number of paths at a time
N = 256
h = T / N

# Estimator variables
sum1 = 0
sum2 = 0
sigbarsum1 = 0
sigbarsum2 = 0
err = 0  # For CVT

# Monte Carlo simulation
for m in range(0, M, M2):
    m2 = min(M2, M - m)
    S = S0 * np.ones(m2, dtype=complex)
    sdot = np.ones(m2, dtype=complex)
    D = np.zeros((N, m2), dtype=complex)
    B = np.zeros((N, m2), dtype=complex)

    # Time evolution
    for n in range(N):
        dW = np.sqrt(h) * np.random.randn(m2)  # Brownian increments
        D[n, :] = 1 + r * h + sig * dW
        B[n, :] = S * dW
        # Forward mode
        sdot *= D[n, :]
        S *= D[n, :]

    # Set sbar(N)
    sbar = np.zeros(m2, dtype=complex)
    for i in range(m2):
        if np.real(S[i]) > K:
            sbar[i] = np.exp(-r * T)
            S[i] = np.exp(-r * T) * S[i]
        else:
            sbar[i] = 0
            S[i] = 0

    # Adjoint recursion
    sigbar = np.zeros(m2, dtype=complex)
    for n in range(N - 1, -1, -1):
        sigbar += sbar * B[n, :]
        sbar = D[n, :] * sbar

    # Check error in dS_N/dS_0
    sbar = np.real(sbar)
    sdot = np.real(sdot)
    err = max(
        err,
        np.max(np.abs(sbar - np.imag(S) / eps)),
        np.max(np.abs(sdot - np.imag(S) / eps)),
    )

    # Use adjoint mode results
    sum1 += np.sum(sbar)
    sum2 += np.sum(sbar**2)
    sigbarsum1 += np.sum(sigbar)
    sigbarsum2 += np.sum(sigbar**2)

# Compute Delta and Vega
delta_approx = sum1 / M  # Delta estimator
sig2 = (sum2 / M - (sum1 / M) ** 2)  # Variance
print(f"Numerical Delta: {delta_approx:.6f} +/- {3 * np.sqrt(sig2 / M):.6f}")

vega_approx = sigbarsum1 / M  # Vega estimator
sig3 = (sigbarsum2 / M - (sigbarsum1 / M) ** 2)  # Variance
print(f"Numerical Vega: {vega_approx:.6f} +/- {3 * np.sqrt(sig3 / M):.6f}")

print(f"Error in dS_N/dS_0 vs CVT result: {err:.6e}")
