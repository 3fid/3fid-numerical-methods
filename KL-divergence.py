import numpy as np

def kl_divergence(P, Q, epsilon=1e-20):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    
    Q = Q + epsilon  # Regularizace
    return np.sum(P * np.log(P / Q))

# Příklad diskrétních pravděpodobností
P = [0.2, 0.5, 0.3]
Q = [0.1, 0.4, 0.0]  # Q má nulovou hodnotu

kl_div = kl_divergence(P, Q)
print(f"KL-divergence: {kl_div}")
