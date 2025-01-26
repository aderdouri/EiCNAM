import numpy as np

def calculate_cva(exposure, default_probabilities, recovery_rate):
    # ...existing code...
    
    # Manual calculation of CVA
    cva = 0.0
    for i in range(len(exposure)):
        # Calculate the incremental CVA for each time step
        incremental_cva = exposure[i] * default_probabilities[i] * (1 - recovery_rate)
        cva += incremental_cva
    
    return cva

# Example usage
exposure = np.array([100, 150, 200])
default_probabilities = np.array([0.01, 0.02, 0.03])
recovery_rate = 0.4

cva = calculate_cva(exposure, default_probabilities, recovery_rate)
print(f"CVA: {cva}")
