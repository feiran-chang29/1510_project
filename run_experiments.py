import numpy as np
from data import generate_keys_independent, generate_keys_correlated, generate_values_onehot
from kernels import relu_kernel, exp_kernel, solu_kernel
from am_model import retrieve, retrieve_raw_weights
from metrics import top1_accuracy, cosine_similarity_batch, estimate_snr

NUM_TRIALS = 5  # 每个 N 重复几次随机实验求平均
# d_k = 32
# d_v = 64  # 可以 >= N
# N_list = [20, 40, 80, 160, 320]  # memory load
d_k = 32
N_list = [50, 100, 200, 400, 800, 1600]
d_v = max(N_list)  # = 320，或者再大一点也行，比如 512
taus = {"relu": None, "exp": 1.0, "solu": 1.0}


def run_single_setting(N, key_mode="correlated", kernel_type="relu"):
    print(f"  [run_single_setting] Start: N={N}, key_mode={key_mode}, kernel={kernel_type}")
    margin_list = []
    # 1. 生成 keys
    if key_mode == "indep":
        K = generate_keys_independent(N, d_k)
    else:
        K = generate_keys_correlated(N, d_k, subspace_dim=8, noise_std=0.1)
    print("    keys generated")
    # 2. 生成 values
    V = generate_values_onehot(N, d_v=d_v)
    print("    values generated")

    # 3. 对每个 key_i 用自己的 key 做 query（也可以加噪声）
    V_true = []
    V_pred = []

    if kernel_type == "relu":
        kernel_fn = relu_kernel
        kernel_kwargs = {}
    elif kernel_type == "exp":
        kernel_fn = exp_kernel
        kernel_kwargs = {"tau": taus["exp"]}
    else:
        kernel_fn = solu_kernel
        kernel_kwargs = {"tau": taus["solu"]}

    print("    starting retrieval loop")
    for i in range(N):
        if i % 20 == 0:
            print(f"      retrieving index i={i}/{N}")
        # q = K[i]  # 或者 q = K[i] + noise
        # q = K[i] + 0.05 * np.random.randn(d_k)
        # q = q / np.linalg.norm(q)
        noise_level = 0.1  
        q = K[i] + noise_level * np.random.randn(d_k)
        q = q / np.linalg.norm(q)


        # v_hat = retrieve(K, V, q, kernel_fn, normalize=True, **kernel_kwargs)
        v_hat = retrieve(K, V, q, kernel_fn, normalize=False)
        V_true.append(V[i])
        V_pred.append(v_hat)

        # --- margin calculation ---
        raw_w = retrieve_raw_weights(K, q, kernel_fn, **kernel_kwargs)
        correct_w = raw_w[i]
        max_wrong = np.max(np.delete(raw_w, i))
        margin_list.append(correct_w - max_wrong)

    print("    retrieval loop done")
    V_true = np.stack(V_true, axis=0)
    V_pred = np.stack(V_pred, axis=0)

    acc = top1_accuracy(V_true, V_pred)
    cos = cosine_similarity_batch(V_true, V_pred)
    snr = estimate_snr(K, kernel_fn, **kernel_kwargs)

    return {
        "acc": acc,
        "cos": cos,
        "snr": snr,
        "margin": np.mean(margin_list)
    }


import pandas as pd

def run_all():
    results = []
    print("[run_all] start running")
    for key_mode in ["indep", "correlated"]:
        print(f"[run_all] key_mode = {key_mode}")
        for kernel_type in ["relu", "exp", "solu"]:
            print(f"    [run_all] kernel = {kernel_type}")
            for N in N_list:
                print(f"        [run_all] N = {N} ... running trial")
                acc_list, cos_list, snr_list, margin_list = [], [], [], []
                for _ in range(NUM_TRIALS):
                    stats = run_single_setting(N, key_mode, kernel_type)
                    acc_list.append(stats["acc"])
                    cos_list.append(stats["cos"])
                    snr_list.append(stats["snr"])
                    margin_list.append(stats["margin"])

                results.append({
                    "key_mode": key_mode,
                    "kernel": kernel_type,
                    "N": N,
                    "acc_mean": np.mean(acc_list),
                    "acc_std": np.std(acc_list),
                    "cos_mean": np.mean(cos_list),
                    "cos_std": np.std(cos_list),
                    "snr_mean": np.mean(snr_list),
                    "snr_std": np.std(snr_list),
                    "margin_mean": np.mean(margin_list),
                    "margin_std": np.std(margin_list),

                })
    df = pd.DataFrame(results)
    df.to_csv("kernel_capacity_results.csv", index=False)
    return df
