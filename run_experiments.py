import numpy as np
import pandas as pd
from data import generate_keys_independent, generate_keys_correlated, generate_values_onehot
from kernels import relu_kernel, exp_kernel, solu_kernel
from am_model import retrieve
from metrics import top1_accuracy, cosine_similarity_batch, estimate_snr

NUM_TRIALS = 5 
d_k = 32
N_list = [80, 160, 240, 320, 400, 480, 560, 640]
d_v = max(N_list) 
taus = {"relu": None, "exp": 1.0, "solu": 1.0}
NOISE_STD_LIST = [0.0, 0.1, 0.2, 0.3]


## ---------------------------------------------
## 1. 定义 Activation Entropy 计算函数
## ---------------------------------------------

def activation_entropy(K, kernel_fn, **kernel_kwargs):
    """
    计算所有 keys 的激活权重向量的平均熵 (Polysemanticity Proxy)。
    使用干净的 key K[i] 作为查询。
    """
    N = K.shape[0]
    entropy_list = []
    
    for i in range(N):
        q = K[i] # 使用干净的 key 作为查询
        w = kernel_fn(K, q, **kernel_kwargs) # (N,)

        # 防止除零和 log(0)，并归一化
        w_sum = w.sum()
        if w_sum <= 1e-8:
            # 如果权重全为 0，熵为 0
            entropy_list.append(0.0) 
            continue
            
        w_norm = w / w_sum
        
        # 避免 np.log(0)
        w_norm = w_norm[w_norm > 1e-8]
        
        # 计算香农熵 (使用自然对数)
        entropy = -np.sum(w_norm * np.log(w_norm))
        entropy_list.append(entropy)
        
    return np.mean(entropy_list)


## ---------------------------------------------
## 2. 修改 run_single_setting 函数
## ---------------------------------------------

def run_single_setting(N, key_mode="correlated", kernel_type="relu", query_noise_std=0.0):
    # 1. 生成 keys
    if key_mode == "indep":
        K = generate_keys_independent(N, d_k)
    else:
        # 使用您代码中的默认参数
        K = generate_keys_correlated(N, d_k, subspace_dim=6, noise_std=0.1) 

    # 2. 生成 values
    V = generate_values_onehot(N, d_v=d_v)

    # 确定 kernel 函数和参数
    if kernel_type == "relu":
        kernel_fn = relu_kernel
        kernel_kwargs = {}
    elif kernel_type == "exp":
        kernel_fn = exp_kernel
        kernel_kwargs = {"tau": taus["exp"]}
    else:
        kernel_fn = solu_kernel
        kernel_kwargs = {"tau": taus["solu"]}

    # 3. 对每个 key_i 用自己的 key 做 query（加噪声）
    V_true = []
    V_pred = []

    for i in range(N):
        q_clean = K[i]
        
        # 引入高斯噪声向量
        noise = query_noise_std * np.random.randn(d_k)
        
        # 构造带噪声的查询 q
        q = q_clean + noise
        
        # 归一化 (保持查询向量长度接近 1)
        q = q / np.linalg.norm(q) 
        
        # 注意：这里 retrieve 默认 normalize=True
        v_hat = retrieve(K, V, q, kernel_fn, normalize=False, **kernel_kwargs) 
        V_true.append(V[i])
        V_pred.append(v_hat)

    V_true = np.stack(V_true, axis=0)
    V_pred = np.stack(V_pred, axis=0)

    # 计算指标
    acc = top1_accuracy(V_true, V_pred)
    cos = cosine_similarity_batch(V_true, V_pred)
    snr = estimate_snr(K, kernel_fn, **kernel_kwargs)
    
    # **新增：计算激活熵**
    # 注意：熵的计算通常不依赖查询噪声，仅依赖 K 和 kernel 函数本身
    entropy = activation_entropy(K, kernel_fn, **kernel_kwargs)

    return {"acc": acc, "cos": cos, "snr": snr, "entropy": entropy} # 返回 'entropy'


## ---------------------------------------------
## 3. 修改 run_all 函数
## ---------------------------------------------

def run_all():
    print("--- Starting Associative Memory Experiments (with Entropy) ---")
    results = []
    total_settings = len(["indep", "correlated"]) * len(["relu", "exp", "solu"]) * len(N_list) * len(NOISE_STD_LIST)
    current_setting = 0
    
    for key_mode in ["indep", "correlated"]:
        for kernel_type in ["relu", "exp", "solu"]:
            for N in N_list:
                for noise_std in NOISE_STD_LIST:
                    
                    current_setting += 1
                    print(f"Running setting {current_setting}/{total_settings}: Mode={key_mode}, Kernel={kernel_type}, N={N}, Noise={noise_std}")

                    acc_list, cos_list, snr_list, entropy_list = [], [], [], [] # 增加 entropy_list
                    
                    for _ in range(NUM_TRIALS):
                        stats = run_single_setting(N, key_mode, kernel_type, query_noise_std=noise_std)
                        acc_list.append(stats["acc"])
                        cos_list.append(stats["cos"])
                        snr_list.append(stats["snr"])
                        entropy_list.append(stats["entropy"]) # 收集 entropy

                    # 记录结果
                    results.append({
                        "key_mode": key_mode,
                        "kernel": kernel_type,
                        "N": N,
                        "noise_std": noise_std,
                        "acc_mean": np.mean(acc_list),
                        "acc_std": np.std(acc_list),
                        "cos_mean": np.mean(cos_list),
                        "cos_std": np.std(cos_list),
                        "snr_mean": np.mean(snr_list),
                        "snr_std": np.std(snr_list),
                        "entropy_mean": np.mean(entropy_list), # 记录 entropy 均值
                        "entropy_std": np.std(entropy_list),  # 记录 entropy 标准差
                    })
    
    df = pd.DataFrame(results)
    filename = "kernel_capacity_results_extended.csv"
    df.to_csv(filename, index=False)
    print(f"--- Experiments Finished. Results saved to {filename} ---")
    return df