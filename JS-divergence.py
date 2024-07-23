import numpy as np

def kl_divergence(P, Q, epsilon=1e-10):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    
    # Přidání malé konstanty epsilon
    P = np.where(P == 0, epsilon, P)
    Q = np.where(Q == 0, epsilon, Q)
    
    return np.sum(P * np.log(P / Q))

def js_divergence(P, Q, epsilon=1e-10):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    M = 0.5 * (P + Q)
    
    return 0.5 * kl_divergence(P, M, epsilon) + 0.5 * kl_divergence(Q, M, epsilon)

# Příklad diskrétních pravděpodobností
P = [0.0, 0.5, 0.4, 0.0, 0.1, 0.0]
Q = [0.6, 0.3, 0.0, 0.0, 0.0, 0.1]

js_div = js_divergence(P, Q, 1e-10)
print(f"Jensen-Shannon divergence: {js_div}")
