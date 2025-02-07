import torch
from Methods.base import PricingMethod

class LongstaffSchwartzMethodBestOf2Assets(PricingMethod):
    def __init__(self, num_paths, num_steps):
        self.num_paths = num_paths
        self.num_steps = num_steps

    def price(self, S0_1, S0_2, K, sigma1, sigma2, T, r, M=12):
        """
        Price the best of two put options using the Longstaff-Schwartz algorithm.

        Args:
            S0_1: Initial price of the first asset.
            S0_2: Initial price of the second asset.
            K: Strike price.
            sigma1: Volatility of the first asset.
            sigma2: Volatility of the second asset.
            T: Time to maturity.
            r: Risk-free rate.
            M: Exercise frequency.

        Returns:
            V: Option value.
        """
        Np = self.num_paths
        NT = self.num_steps
        dt = T / torch.tensor(NT, dtype=torch.float32)
        sqrt_dt = torch.sqrt(dt)

        # Simulate paths for both assets
        Z1 = torch.randn(Np, NT)
        Z2 = torch.randn(Np, NT)
        Sp1 = torch.zeros(Np, NT, dtype=torch.float32)
        Sp2 = torch.zeros(Np, NT, dtype=torch.float32)
        Sp1[:, 0] = S0_1
        Sp2[:, 0] = S0_2

        for t in range(1, NT):
            previous_step1 = Sp1[:, t - 1].clone()
            previous_step2 = Sp2[:, t - 1].clone()
            Sp1[:, t] = previous_step1 * torch.exp((r - 0.5 * sigma1**2) * dt + sigma1 * sqrt_dt * Z1[:, t])
            Sp2[:, t] = previous_step2 * torch.exp((r - 0.5 * sigma2**2) * dt + sigma2 * sqrt_dt * Z2[:, t])

        # Initialize cash flows
        cash_flow = torch.maximum(K - torch.minimum(Sp1[:, -1], Sp2[:, -1]), torch.tensor(0.0, dtype=torch.float32))
        discount_factor = torch.exp(-r * dt)

        # Backward induction
        cash_flow = cash_flow.clone()
        for t in range(NT - 2, 0, -M):
            in_the_money = torch.minimum(Sp1[:, t], Sp2[:, t]) < K
            itm_indices = torch.where(in_the_money)[0]

            if len(itm_indices) > 0:
                X1 = Sp1[itm_indices, t]
                X2 = Sp2[itm_indices, t]
                Y = cash_flow[itm_indices] * discount_factor.clone()

                # Regression to approximate continuation value
                A = torch.stack([torch.ones_like(X1), X1, X1**2, X2, X2**2], dim=1)
                coeffs = torch.linalg.lstsq(A, Y).solution

                continuation_value = coeffs[0] + coeffs[1] * X1 + coeffs[2] * X1**2 + coeffs[3] * X2 + coeffs[4] * X2**2

                exercise_value = K - torch.minimum(X1, X2)

                exercise = exercise_value > continuation_value
                exercise_indices = itm_indices[exercise]

                cash_flow = cash_flow.clone()
                cash_flow[exercise_indices] = exercise_value[exercise]

            cash_flow = cash_flow * discount_factor.clone()

        # Final option value
        V = cash_flow.mean() * torch.exp(-r * dt)
        return V

    def calculate_greeks(self, S0_1, S0_2, K, sigma1, sigma2, T, r, M=12):
        """
        Calculate sensitivities (Delta, Vega, Rho, Theta) for two assets using automatic differentiation.

        Args:
            S0_1: Initial price of the first asset.
            S0_2: Initial price of the second asset.
            K: Strike price.
            sigma1: Volatility of the first asset.
            sigma2: Volatility of the second asset.
            T: Time to maturity.
            r: Risk-free rate.
            M: Exercise frequency.

        Returns:
            sensitivities: Dictionary containing Delta1, Delta2, Vega1, Vega2, Rho, Theta.
        """
        S0_1_t = torch.tensor(S0_1, requires_grad=True, dtype=torch.float32)
        S0_2_t = torch.tensor(S0_2, requires_grad=True, dtype=torch.float32)
        sigma1_t = torch.tensor(sigma1, requires_grad=True, dtype=torch.float32)
        sigma2_t = torch.tensor(sigma2, requires_grad=True, dtype=torch.float32)
        r_t = torch.tensor(r, requires_grad=True, dtype=torch.float32)
        T_t = torch.tensor(T, requires_grad=True, dtype=torch.float32)

        # Enable anomaly detection
        with torch.autograd.set_detect_anomaly(True):
            # Compute option value
            V = self.price(S0_1_t, S0_2_t, K, sigma1_t, sigma2_t, T_t, r_t, M)

            # Compute gradients
            V.backward()

        delta1 = S0_1_t.grad.item()
        delta2 = S0_2_t.grad.item()
        vega1 = sigma1_t.grad.item()
        vega2 = sigma2_t.grad.item()
        rho = r_t.grad.item()
        theta = T_t.grad.item()

        return {
            "Delta1": delta1,
            "Delta2": delta2,
            "Vega1": vega1,
            "Vega2": vega2,
            "Rho": rho,
            "Theta": theta
        }
