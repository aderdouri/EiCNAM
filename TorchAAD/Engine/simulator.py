import torch
def simulate_process(process, S0, T, steps, n_paths):
    """
    Simulates paths for a given stochastic process.

    Args:
        process: An instance of a subclass of StochasticProcess.
        S0: Initial value (scalar or tensor).
        T: Total time (float).
        steps: Number of time steps (int).
        n_paths: Number of simulated paths (int).

    Returns:
        A tensor of shape (steps, n_paths) containing simulated paths.
    """
    dt = T / steps
    S = torch.full((steps, n_paths), S0, dtype=torch.float32, device=process.device)
    
    # Generate Brownian increments
    dW = torch.randn(steps - 1, n_paths, dtype=torch.float32, device=process.device) * torch.sqrt(torch.tensor(dt, device=process.device))
    
    # Simulate paths iteratively
    for t in range(1, steps):
        S[t] = process.evolve(S[t - 1], dt, dW[t - 1])
    
    return S.T