# visualize.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# ================================
# 0. Results Loader
# ================================
def load_results(csv_path="results/kernel_capacity_results.csv"):
    """
    Load the results CSV produced by run_experiments.py
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df


# ================================
# 1. Prepare Plot Folder
# ================================
def ensure_plot_dir():
    if not os.path.exists("plots"):
        os.makedirs("plots")


# ================================
# 2. Colors for kernels
# ================================
KERNEL_COLORS = {
    "relu": "#1f77b4",   # blue
    "solu": "#2ca02c",   # green
    "exp":  "#ff7f0e",   # orange
}

# ================================
# 3. Accuracy vs N
# ================================
def plot_accuracy(df, save_prefix="plots/accuracy_vs_n"):
    """
    Plot retrieval accuracy vs N, separately for key_mode = indep / correlated.
    df: DataFrame from load_results()
    """
    ensure_plot_dir()

    for key_mode in sorted(df["key_mode"].unique()):
        sub = df[df["key_mode"] == key_mode]

        plt.figure(figsize=(6, 4))
        for kernel in ["relu", "solu", "exp"]:
            temp = sub[sub["kernel"] == kernel].sort_values("N")
            if temp.empty:
                continue

            Ns = temp["N"].values
            mean = temp["acc_mean"].values
            std = temp["acc_std"].values

            plt.errorbar(
                Ns,
                mean,
                yerr=std,
                marker="o",
                capsize=3,
                label=kernel.upper(),
                color=KERNEL_COLORS.get(kernel, None),
            )

        plt.xlabel("Memory load N")
        plt.ylabel("Top-1 accuracy")
        plt.title(f"Retrieval Accuracy vs N ({key_mode})")
        plt.ylim(0.0, 1.05)
        plt.legend()
        plt.tight_layout()

        out_path = f"{save_prefix}_{key_mode}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[plot_accuracy] Saved: {out_path}")


# ================================
# 4. SNR vs N
# ================================
def plot_snr(df, save_prefix="plots/snr_vs_n"):
    """
    Plot SNR vs N, separately for key_mode = indep / correlated.
    df: DataFrame from load_results()
    """
    ensure_plot_dir()

    for key_mode in sorted(df["key_mode"].unique()):
        sub = df[df["key_mode"] == key_mode]

        plt.figure(figsize=(6, 4))
        for kernel in ["relu", "solu", "exp"]:
            temp = sub[sub["kernel"] == kernel].sort_values("N")
            if temp.empty:
                continue

            Ns = temp["N"].values
            mean = temp["snr_mean"].values
            std = temp["snr_std"].values

            plt.errorbar(
                Ns,
                mean,
                yerr=std,
                marker="o",
                capsize=3,
                label=kernel.upper(),
                color=KERNEL_COLORS.get(kernel, None),
            )

        plt.xlabel("Memory load N")
        plt.ylabel("SNR (signal / noise)")
        plt.title(f"SNR vs N ({key_mode})")
        plt.legend()
        plt.tight_layout()

        out_path = f"{save_prefix}_{key_mode}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[plot_snr] Saved: {out_path}")

# ================================
# 5. Margin vs N
# ================================
def plot_margin(df, save_prefix="plots/margin_vs_n"):
    """
    Plot margin (w_correct - max w_other) vs N.
    This is much more sensitive than accuracy.
    """
    ensure_plot_dir()

    for key_mode in sorted(df["key_mode"].unique()):
        sub = df[df["key_mode"] == key_mode]

        plt.figure(figsize=(6, 4))
        for kernel in ["relu", "solu", "exp"]:
            temp = sub[sub["kernel"] == kernel].sort_values("N")
            if temp.empty:
                continue

            Ns = temp["N"].values
            mean = temp["margin_mean"].values
            std = temp["margin_std"].values

            plt.errorbar(
                Ns,
                mean,
                yerr=std,
                marker="o",
                capsize=3,
                label=kernel.upper(),
                color=KERNEL_COLORS.get(kernel, None),
            )

        plt.xlabel("Memory load N")
        plt.ylabel("Retrieval margin")
        plt.title(f"Margin vs N ({key_mode})")
        plt.legend()
        plt.tight_layout()

        out_path = f"{save_prefix}_{key_mode}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[plot_margin] Saved: {out_path}")

# ================================
# 6. Activation Heatmap (Polysemanticity Visualization)
# ================================
from am_model import retrieve_raw_weights
from kernels import relu_kernel, exp_kernel, solu_kernel


def plot_activation_maps(K, q=None, save_prefix="plots/activation_map"):
    """
    Visualize the weight distribution (kernel(q, k_i)) as heatmaps
    for ReLU, SoLU, Exp kernels.

    K: (N, d_k) matrix of keys
    q: query vector; if None, use K[0] (self-retrieval)
    """

    ensure_plot_dir()

    N = K.shape[0]
    if q is None:
        q = K[0]  # self-query (common baseline)

    # Kernels to compare
    kernel_fns = {
        "relu": relu_kernel,
        "solu": solu_kernel,
        "exp": exp_kernel,
    }

    for name, kernel_fn in kernel_fns.items():
        # Compute raw weights (no normalization)
        if name == "relu":
            w = retrieve_raw_weights(K, q, kernel_fn)
        else:
            w = retrieve_raw_weights(K, q, kernel_fn, tau=1.0)

        # w = retrieve_raw_weights(K, q, kernel_fn, tau=1.0)
        w = w.reshape(1, -1)  # reshape to [1 x N] for heatmap

        plt.figure(figsize=(10, 1.8))
        sns.heatmap(
            w,
            cmap="viridis",
            cbar=True,
            xticklabels=False,
            yticklabels=False
        )
        plt.title(f"Activation Heatmap — {name.upper()} Kernel")
        plt.tight_layout()

        out_path = f"{save_prefix}_{name}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[plot_activation_maps] Saved: {out_path}")

# ================================
# 7. Weight Distribution Plot
# ================================

def plot_weight_distribution(K, q=None, save_path="plots/weight_distribution.png"):
    """
    Plot sorted raw weights for ReLU, SoLU, Exp kernels.
    Shows 'fat tail' (ReLU) vs 'sharp peak' (Exp).
    """

    ensure_plot_dir()

    N = K.shape[0]
    if q is None:
        q = K[0]  # self-query

    kernel_fns = {
        "relu": relu_kernel,
        "solu": solu_kernel,
        "exp": exp_kernel,
    }

    plt.figure(figsize=(7, 5))

    for name, kernel_fn in kernel_fns.items():
        # w = retrieve_raw_weights(K, q, kernel_fn, tau=1.0)
        if name == "relu":
            w = retrieve_raw_weights(K, q, kernel_fn)
        else:
            w = retrieve_raw_weights(K, q, kernel_fn, tau=1.0)
        w_sorted = np.sort(w)[::-1]  # descending

        plt.plot(
            w_sorted,
            label=name.upper(),
            linewidth=2,
            color=KERNEL_COLORS.get(name, None),
        )

    plt.xlabel("Key index (sorted by activation)")
    plt.ylabel("Raw activation weight")
    plt.title("Sorted Weight Distribution (Polysemanticity Comparison)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_weight_distribution] Saved: {save_path}")

# ================================
# 8. Capacity Frontier (Sharpness Sweep)
# ================================
from am_model import retrieve
from data import generate_keys_correlated, generate_values_onehot

def measure_accuracy_for_sharpness(N, d_k, d_v, tau, kernel_type="exp",
                                   subspace_dim=4, noise_level=0.2, trials=3):
    """
    Measure average accuracy at a given sharpness tau for a fixed memory load N.
    """
    acc_list = []

    # choose kernel
    if kernel_type == "exp":
        kernel_fn = exp_kernel
        kernel_kwargs = {"tau": tau}
    elif kernel_type == "solu":
        kernel_fn = solu_kernel
        kernel_kwargs = {"tau": tau}
    else:
        raise ValueError("kernel_type must be exp or solu")

    for _ in range(trials):
        # correlated keys cause interference
        K = generate_keys_correlated(N, d_k, subspace_dim=subspace_dim)

        # values are one-hot
        V = generate_values_onehot(N, d_v)

        correct = 0
        for i in range(N):
            q = K[i] + noise_level * np.random.randn(d_k)
            v_hat = retrieve(K, V, q, kernel_fn, normalize=False, **kernel_kwargs)
            if np.argmax(v_hat) == i:
                correct += 1

        acc_list.append(correct / N)

    return np.mean(acc_list)


def plot_capacity_frontier(
    d_k=32,
    d_v=64,
    N_list=[50, 100, 200, 400, 800, 1200],
    tau_list=np.linspace(0.2, 3.0, 12),
    kernel_type="solu",
    save_path="plots/capacity_frontier.png"
):
    """
    Sweep sharpness (tau) and compute max N such that accuracy >= 0.75.
    Produces a capacity frontier curve.
    """
    ensure_plot_dir()

    capacity_values = []

    for tau in tau_list:
        max_N = 0
        for N in N_list:
            acc = measure_accuracy_for_sharpness(
                N, d_k, d_v, tau,
                kernel_type=kernel_type,
                subspace_dim=4,
                noise_level=0.2,
                trials=3
            )
            if acc >= 0.75:   # capacity threshold
                max_N = N
        capacity_values.append(max_N)
        print(f"[capacity] tau={tau:.2f} → capacity N={max_N}")

    # plot
    plt.figure(figsize=(7, 5))
    plt.plot(
        tau_list,
        capacity_values,
        marker="o",
        linewidth=2,
        label=f"{kernel_type.upper()} kernel"
    )

    plt.xlabel("Kernel sharpness τ")
    plt.ylabel("Capacity N* (accuracy ≥ 0.75)")
    plt.title("Capacity Frontier: Sharpness vs Maximum Storable N")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_capacity_frontier] Saved: {save_path}")

# ================================
# 9. One-click pipeline: generate all plots
# ================================
def generate_all_plots(csv_path="kernel_capacity_results.csv"):
    """
    Run all visualization functions.
    Make sure run_experiments.py has already produced results CSV.
    """

    print("\n[generate_all_plots] Loading results CSV...")
    df = load_results(csv_path)
    ensure_plot_dir()

    print("[generate_all_plots] Plotting Accuracy curves...")
    plot_accuracy(df)

    print("[generate_all_plots] Plotting SNR curves...")
    plot_snr(df)

    print("[generate_all_plots] Plotting Margin curves...")
    plot_margin(df)

    print("[generate_all_plots] Generating activation maps...")
    K = generate_keys_correlated(N=200, d_k=32, subspace_dim=4)
    plot_activation_maps(K)

    print("[generate_all_plots] Plotting weight distributions...")
    plot_weight_distribution(K)

    print("[generate_all_plots] Sweeping sharpness to compute capacity frontier...")
    plot_capacity_frontier(kernel_type="solu")
    plot_capacity_frontier(kernel_type="exp",
                           save_path="plots/capacity_frontier_exp.png")

    print("\n🎉 All plots generated and saved in /plots folder!")

