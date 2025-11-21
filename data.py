# data.py
import numpy as np

def generate_keys_independent(N, d_k):
    # 高维高斯，归一化
    K = np.random.randn(N, d_k)
    K = K / np.linalg.norm(K, axis=1, keepdims=True)
    return K  # shape: (N, d_k)

def generate_keys_correlated(N, d_k, subspace_dim=8, noise_std=0.1):
    # 在一个低维子空间里采样，再加噪声 → keys 高度相关
    base = np.random.randn(subspace_dim, d_k)  # 子空间基
    coeffs = np.random.randn(N, subspace_dim)
    K = coeffs @ base  # (N, d_k)
    K = K + noise_std * np.random.randn(N, d_k)
    K = K / np.linalg.norm(K, axis=1, keepdims=True)
    return K

def generate_values_onehot(N, d_v=None):
    if d_v is None:
        d_v = N  # 简单方案：每个 key 一个独立 one-hot
    V = np.zeros((N, d_v))
    idx = np.random.choice(d_v, size=N, replace=False) if d_v >= N else np.arange(N)
    V[np.arange(N), idx] = 1.0
    return V  # (N, d_v), 适合做 top-1 accuracy
