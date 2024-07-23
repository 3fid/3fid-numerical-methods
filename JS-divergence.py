import numpy as np

def kl_divergence(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    return np.sum(P * np.log(P / Q))

def js_divergence(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)

# Příklad diskrétních pravděpodobností
P = [0.2, 0.5, 0.3]
Q = [0.1, 0.4, 0.5]

js_div = js_divergence(P, Q)
print(f"Jensen-Shannon divergence: {js_div}")
