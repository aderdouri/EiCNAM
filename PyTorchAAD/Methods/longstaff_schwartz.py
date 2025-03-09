import torch
from .base import PricingMethod
from Engine.stochastic_process import IntensityProcess

# Function to compute Hermite polynomial basis functions up to order 2 (3 basis functions: H0, H1, H2)
def hermite_basis(X, order=2):
    """
    Compute the Hermite polynomial basis functions up to a given order.
    """
    H = [torch.ones_like(X), 2 * X]  # H_0(x) = 1, H_1(x) = 2x

    for n in range(2, order + 1):
        Hn = 2 * X * H[n - 1] - 2 * (n - 1) * H[n - 2]  # Recurrence relation
        H.append(Hn)

    return torch.stack(H, dim=1)  # Stack as feature matrix
class LongstaffSchwartzMethod(PricingMethod):
    def __init__(self, num_paths=500000, num_steps=1000):
        self.num_paths = num_paths
        self.num_steps = num_steps

    def price(self, S0, K, sigma, T, r, M=3, use_cir=False, cir_params=None):
        """
        Longstaff-Schwartz algorithm implemented in PyTorch.

        Args:
            S0: Initial asset price.
            K: Strike price.
            sigma: Volatility.
            T: Time to maturity.
            r: Risk-free rate.
            M: Exercise frequency.
            use_cir: Whether to use the CIR intensity model.
            cir_params: Parameters for the CIR model (mu, k, nu).

        Returns:
            V: Option value.
        """

        print(f"Running Longstaff-Schwartz algorithm with {self.num_paths} paths, {self.num_steps} steps and M={M}")
        Np = self.num_paths
        NT = self.num_steps
        dt = T / torch.tensor(NT, dtype=torch.float32)  # Ensure dt is a tensor
        sqrt_dt = torch.sqrt(dt)

        # Simulate paths
        Z = torch.randn(Np, NT)
        Sp = torch.zeros(Np, NT, dtype=torch.float32)
        Sp[:, 0] = S0

        for t in range(1, NT):
            previous_step = Sp[:, t - 1].clone()  # Avoid modifying previous step
            Sp[:, t] = previous_step * torch.exp((r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z[:, t])

        if use_cir and cir_params:
            mu, k, nu = cir_params
            lambda_0 = 1.0
            lambda_t = torch.tensor(lambda_0, requires_grad=True)
            intensity_process = Process(mu=mu, sigma=0.0, k=k, nu=nu)
            mc = MonteCarloMethod(intensity_process, lambda_t, T, Np, NT)
            lambdas_paths = mc.simulate()

            dt_step = T / NT
            integrated_intensity = torch.sum(lambdas_paths.T[:, :-1] * dt_step, dim=1)
            survival_probs = torch.exp(-integrated_intensity).mean()
        else:
            survival_probs = torch.tensor(1.0)

        # Initialize cash flows
        cash_flow = torch.maximum(K - Sp[:, -1], torch.tensor(0.0, dtype=torch.float32))
        discount_factor = torch.exp(-r * dt)

        # Backward induction
        cash_flow = cash_flow.clone()  # Ensure no inplace modification
        for t in range(NT - 2, 0, -M):  # Adjust the step to M
            in_the_money = Sp[:, t] < K
            itm_indices = torch.where(in_the_money)[0]

            if len(itm_indices) > 0:
                X = Sp[itm_indices, t]
                Y = cash_flow[itm_indices] * discount_factor.clone()

                # Regression to approximate continuation value
                # A = torch.stack([torch.ones_like(X), X, X**2, X**3], dim=1)
                # coeffs = torch.linalg.lstsq(A, Y).solution
                # continuation_value = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2 + coeffs[3] * X**3


                # Compute Hermite polynomial basis for regression
                A = hermite_basis(X, order=2)  # Using 3 basis functions (H0, H1, H2)
                # Solve Least Squares Regression: A * coeffs ≈ Y
                coeffs = torch.linalg.lstsq(A, Y).solution
                # Compute continuation value using Hermite polynomials
                continuation_value = coeffs[0] + coeffs[1] * (2 * X) + coeffs[2] * (4 * X**2 - 2)  # H0, H1, H2


                exercise_value = K - X

                exercise = exercise_value > continuation_value
                exercise_indices = itm_indices[exercise]

                cash_flow = cash_flow.clone()  # Avoid inplace modification
                cash_flow[exercise_indices] = exercise_value[exercise]

            cash_flow = cash_flow * discount_factor.clone()  # Ensure no inplace modification

        # Final option value
        V = cash_flow.mean() * torch.exp(-r * dt) * survival_probs
        print(f"Option value: {V.item()}")
        return V

    def calculate_greeks(self, S0, K, sigma, T, r, M=12, use_cir=False, cir_params=None):
        """
        Calculate sensitivities (Delta, Vega, Rho, Theta) using automatic differentiation.

        Args:
            S0: Initial asset price.
            K: Strike price.
            sigma: Volatility.
            T: Time to maturity.
            r: Risk-free rate.
            M: Exercise frequency.
            use_cir: Whether to use the CIR intensity model.
            cir_params: Parameters for the CIR model (mu, k, nu).

        Returns:
            sensitivities: Dictionary containing Delta, Vega, Rho, Theta.
        """
        S0_t = torch.tensor(S0, requires_grad=True, dtype=torch.float32)
        sigma_t = torch.tensor(sigma, requires_grad=True, dtype=torch.float32)
        r_t = torch.tensor(r, requires_grad=True, dtype=torch.float32)
        T_t = torch.tensor(T, requires_grad=True, dtype=torch.float32)

        # Enable anomaly detection
        with torch.autograd.set_detect_anomaly(True):
            # Compute option value
            V = self.price(S0_t, K, sigma_t, T_t, r_t, M, use_cir, cir_params)

            # Compute gradients
            V.backward()

        delta = S0_t.grad.item()
        vega = sigma_t.grad.item()
        rho = r_t.grad.item()
        theta = T_t.grad.item()

        return {
            "Delta": delta,
            "Vega": vega,
            "Rho": rho,
            "Theta": theta
        }
