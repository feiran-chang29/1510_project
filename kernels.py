import numpy as np

def dot_kernel(K, q):
    # K: (N, d_k), q: (d_k,)
    return K @ q  # (N,)

def relu_kernel(K, q):
    s = K @ q
    return np.maximum(0.0, s)

def exp_kernel(K, q, tau=0.5):
    s = K @ q / tau
    # 防止数值爆炸，可以 clip 一下
    s = np.clip(s, -20, 20)
    return np.exp(s)

# def solu_kernel(K, q, tau=1.0):
#     s = K @ q
#     s_over_tau = np.clip(s / tau, -20, 20)
#     return s * np.exp(s_over_tau)

def solu_kernel(K, q, tau=1.0):
    """
    A SoLU-like positive kernel:
    smooth, between ReLU and exp.
    """
    s = K @ q / tau              # (N,)
    s = np.clip(s, -20, 20)
    # softplus: log(1 + exp(s))  >= 0, smooth
    return np.log1p(np.exp(s))
