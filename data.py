# data.py
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Pairwise cosine similarity
# -------------------------------
def pairwise_cosine(K):
    """
    K: (N, d_k)
    return: all pairwise cosines (upper triangle, excluding diag)
    """
    # Gram matrix G_ij = k_i^T k_j
    G = K @ K.T    # shape (N, N)
    # Extract upper triangle
    idx = np.triu_indices(len(K), k=1)
    return G[idx]


def generate_keys_independent(N, d_k):
    # 高维高斯，归一化
    K = np.random.randn(N, d_k)
    K = K / np.linalg.norm(K, axis=1, keepdims=True)
    return K  # shape: (N, d_k)

def generate_keys_correlated(N, d_k, subspace_dim=4, noise_std=0.1, seed=None):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((d_k, subspace_dim))  # d_k x subspace_dim
    # Orthogonalize base columns
    Q, _ = np.linalg.qr(base)
    base = Q[:, :subspace_dim]  # d_k x subspace_dim
    coeffs = rng.standard_normal((N, subspace_dim))
    K = coeffs @ base.T  # (N, d_k)
    K = K + noise_std * rng.standard_normal((N, d_k))
    norms = np.linalg.norm(K, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    K = K / norms
    return K.astype(np.float32)


def generate_values_onehot(N, d_v=None):
    if d_v is None:
        d_v = N
    V = np.zeros((N, d_v))
    if d_v >= N:
        # 每个样本用一个不重复的列
        idx = np.random.choice(d_v, size=N, replace=False)
    else:
        # 真的出现 d_v < N，也不会炸，只是会有重复 label
        idx = np.random.choice(d_v, size=N, replace=True)
    V[np.arange(N), idx] = 1.0
    return V

# if __name__ == "__main__":
#     N = 300
#     d_k = 32
    
#     K_ind = generate_keys_independent(N, d_k)
#     K_cor = generate_keys_correlated(N, d_k, subspace_dim=4, noise_std=0.1)
    
#     cos_ind = pairwise_cosine(K_ind)
#     cos_cor = pairwise_cosine(K_cor)
    
#     print("Independent keys:")
#     print("  mean cos =", cos_ind.mean())
#     print("  std cos  =", cos_ind.std())
    
#     print("\nCorrelated keys:")
#     print("  mean cos =", cos_cor.mean())
#     print("  std cos  =", cos_cor.std())
    
#     # -------------------------
#     # Plot histograms
#     # -------------------------
#     plt.figure(figsize=(8,5))
#     bins = np.linspace(-1.0, 1.0, 80)
    
#     plt.hist(cos_ind, bins=bins, alpha=0.6, label="Independent", density=True)
#     plt.hist(cos_cor, bins=bins, alpha=0.6, label="Correlated", density=True)
    
#     plt.title("Pairwise Cosine Similarity Distribution")
#     plt.xlabel("cos(k_i, k_j)")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.grid(alpha=0.3)
    
#     plt.show()
