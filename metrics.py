import numpy as np

def top1_accuracy(V_true, V_pred):
    # V_true: (num_queries, d_v)
    # V_pred: (num_queries, d_v)
    true_idx = V_true.argmax(axis=1)
    pred_idx = V_pred.argmax(axis=1)
    return (true_idx == pred_idx).mean()

def cosine_similarity_batch(V_true, V_pred):
    num = (V_true * V_pred).sum(axis=1)
    denom = (np.linalg.norm(V_true, axis=1) * np.linalg.norm(V_pred, axis=1) + 1e-8)
    return (num / denom).mean()

def estimate_snr(K, kernel_fn, **kernel_kwargs):
    N = K.shape[0]
    snr_list = []
    for i in range(N):
        q = K[i]
        w = kernel_fn(K, q, **kernel_kwargs)  # (N,)
        signal = w[i] ** 2
        noise = (np.delete(w, i) ** 2).sum()
        if noise == 0:
            snr = np.inf
        else:
            snr = signal / noise
        snr_list.append(snr)
    return np.mean(snr_list)
