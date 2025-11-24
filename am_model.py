from kernels import relu_kernel, exp_kernel, solu_kernel
import numpy as np

def retrieve(K, V, q, kernel_fn, normalize=True, **kernel_kwargs):
    """
    K: (N, d_k)
    V: (N, d_v)
    q: (d_k,)
    kernel_fn: relu_kernel / exp_kernel / solu_kernel
    """
    w = kernel_fn(K, q, **kernel_kwargs)  # (N,)
    if normalize:
        if w.sum() == 0:
            # 防止全 0
            return np.zeros(V.shape[1])
        w = w / (w.sum() + 1e-8)
    # 加权和
    return w @ V  # (d_v,)

def build_S(K, V, phi=None):
    """
    如果有特征映射 phi，就传入；否则 phi = identity
    """
    if phi is None:
        phi = lambda x: x
    PhiK = np.stack([phi(k) for k in K], axis=0)  # (N, d_phi)
    # S = sum_i v_i phi(k_i)^T，相当于 V^T @ PhiK
    S = V.T @ PhiK
    return S  # (d_v, d_phi)

def retrieve_raw_weights(K, q, kernel_fn, **kwargs):
    """
    Returns raw kernel weights w_i before normalization & V combination.
    K: (N, d_k)
    q: (d_k,)
    """
    N = K.shape[0]
    w = np.zeros(N)
    for i in range(N):
        w[i] = kernel_fn(q, K[i], **kwargs)
    return w

