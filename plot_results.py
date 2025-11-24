import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os # 确保 os 被导入
from data import generate_keys_independent, generate_keys_correlated
from kernels import relu_kernel, exp_kernel, solu_kernel # 假设 kernels.py 包含这些函数

# --- 全局设置和数据加载 ---

# ... (数据加载和全局设置部分保持不变) ...

sns.set(style="whitegrid")

# --- 全局设置和数据加载 ---

# 假设您的 CSV 文件名已更新
try:
    df = pd.read_csv("kernel_capacity_results_extended.csv")
except FileNotFoundError:
    print("Error: 'kernel_capacity_results_extended.csv' not found. Please run the experiments first.")
    df = pd.DataFrame() 

# 获取实验中使用的噪声值列表
if 'noise_std' in df.columns and not df.empty:
    NOISE_STD_LIST = sorted(df['noise_std'].unique())
    # 假设 TEST_NOISE_STD 是列表中的最大值，用于容量曲线
    TEST_NOISE_STD = NOISE_STD_LIST[-1] if len(NOISE_STD_LIST) > 1 else 0.0
else:
    NOISE_STD_LIST = [0.0]
    TEST_NOISE_STD = 0.0

sns.set(style="whitegrid")

# ----------------------------------------------------
# 1 & 2. Retrieval Accuracy vs N (Capacity Plots)
# ----------------------------------------------------

def plot_accuracy_capacity():
    """ 
    绘制 Top-1 Accuracy 结果，主要关注高噪声/高干扰下的容量曲线。
    对应要求 1 (correlated) 和 2 (independent)。
    """
    for key_mode in ["indep", "correlated"]:
        plt.figure(figsize=(7, 5))
        
        # 筛选出要绘制的噪声级别。假设我们用最高的噪声来测试鲁棒容量。
        # 如果 NOISE_STD_LIST 只有一个值（0.0），就用 0.0
        target_noise = TEST_NOISE_STD 

        sub = df[(df["key_mode"] == key_mode) & (df["noise_std"] == target_noise)]

        for kernel in ["relu", "exp", "solu"]:
            temp = sub[sub["kernel"] == kernel]
            
            # 确保数据不为空
            if not temp.empty:
                plt.errorbar(
                    temp["N"], temp["acc_mean"],
                    yerr=temp["acc_std"],
                    label=kernel,
                    marker='o',
                    capsize=3,
                    linestyle='-'
                )
            
        plt.title(f"Retrieval Accuracy vs N [{key_mode}] ($\sigma_q$={target_noise:.2f})")
        plt.xlabel("N (memory load)")
        plt.ylabel("Top-1 Accuracy")
        
        # 在 correlated 图上标注 Capacity Threshold
        if key_mode == "correlated":
            plt.axhline(y=0.7, color='gray', linestyle='--', label="Capacity Threshold (Acc=0.7)", zorder=0)
        
        plt.ylim(-0.05, 1.05) # 保证 Y 轴范围完整
        plt.legend()
        plt.savefig(f"plots/acc_capacity_{key_mode}.png")
        plt.show()

# ----------------------------------------------------
# 3. SNR vs N (线性 Y 轴和定义标注)
# ----------------------------------------------------

def plot_snr_vs_n():
    """ 
    绘制 Signal-to-Noise Ratio (SNR) 结果，使用线性Y轴并标注定义。
    对应要求 3。
    """
    for key_mode in ["indep", "correlated"]:
        plt.figure(figsize=(7, 5))
        
        # SNR 估计使用无噪声基线 (noise_std == 0.0)
        sub = df[(df["key_mode"] == key_mode) & (df["noise_std"] == 0.0)] 

        for kernel in ["relu", "exp", "solu"]:
            temp = sub[sub["kernel"] == kernel]
            if not temp.empty:
                plt.errorbar(
                    temp["N"], temp["snr_mean"],
                    yerr=temp["snr_std"],
                    label=kernel,
                    marker='o',
                    capsize=3
                )

        # plt.yscale("linear") # 默认为线性，无需设置
        plt.title(f"SNR vs N [{key_mode}] (Noise $\sigma_q$=0.0)")
        plt.xlabel("N (memory load)")
        plt.ylabel("SNR (Signal/Noise)")
        
        # 添加 SNR 的数学定义
        plt.text(0.5, 0.95, 
                 r'$\text{SNR} = \frac{\kappa^2(k_i, k_i)}{\sum_{j \neq i} \kappa^2(k_j, k_i)}$', 
                 transform=plt.gca().transAxes, # 坐标转换为图轴比例
                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.legend()
        plt.savefig(f"plots/snr_{key_mode}_linear.png")
        plt.show()

# ----------------------------------------------------
# 4. Activation Entropy / Polysemanticity vs N
# ----------------------------------------------------

def plot_entropy_vs_n():
    """ 
    绘制 Activation Entropy 结果，作为 Polysemanticity 指标。
    对应要求 4。
    """
    # 检查 CSV 中是否有 entropy 数据
    if 'entropy_mean' not in df.columns:
        print("Entropy data not available in CSV. Skipping entropy plot.")
        return

    for key_mode in ["indep", "correlated"]:
        plt.figure(figsize=(7, 5))
        
        # 熵的计算与查询噪声无关，使用无噪声基线数据
        sub = df[(df["key_mode"] == key_mode) & (df["noise_std"] == 0.0)]

        for kernel in ["relu", "exp", "solu"]:
            temp = sub[sub["kernel"] == kernel]
            if not temp.empty:
                plt.errorbar(
                    temp["N"], temp["entropy_mean"],
                    yerr=temp["entropy_std"],
                    label=kernel,
                    marker='o',
                    capsize=3
                )

        plt.title(f"Activation Entropy vs N [{key_mode}] (Polysemanticity Proxy)")
        plt.xlabel("N (memory load)")
        plt.ylabel("Average Activation Entropy")
        
        # 添加期望的解释（高熵 -> 激活分散 -> 更多 Polysemanticity）
        plt.text(0.05, 0.95, 
                 "Higher Entropy $\\rightarrow$ More Polysemantic (Dense Activation)", 
                 transform=plt.gca().transAxes, 
                 fontsize=9, verticalalignment='top')
                 
        plt.legend()
        plt.savefig(f"plots/entropy_{key_mode}.png")
        plt.show()

# ----------------------------------------------------
# 6. Activation Heatmap (Polysemanticity Visualization)
# ----------------------------------------------------

# 辅助函数：模拟 retrieve_raw_weights，因为它可能在 am_model.py 中
def _retrieve_raw_weights_local(K, q, kernel_fn, **kernel_kwargs):
    """
    计算原始权重向量 w = [kernel(q, k_i), ...]
    """
    # 确保 q 是二维数组以便广播
    if q.ndim == 1:
        q = q[np.newaxis, :]
        
    # kernel_fn 应该返回一个 (N,) 形状的向量
    w = kernel_fn(K, q.squeeze(), **kernel_kwargs) 
    return w.squeeze()


def ensure_plot_dir():
    if not os.path.exists("plots"):
        os.makedirs("plots")


def plot_activation_maps(K, d_k, q=None, save_prefix="plots/activation_map"):
    """
    Visualize the weight distribution (kernel(q, k_i)) as heatmaps
    for ReLU, SoLU, Exp kernels.

    K: (N, d_k) matrix of keys
    q: query vector; if None, use K[0] (self-retrieval)
    d_k: dimensionality (用于 Exp kernel 的 tau=sqrt(d_k) 缩放)
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
    
    # 使用 tau = sqrt(d_k) 进行 Exp 和 SoLU 缩放（与标准 Attention 一致）
    tau_exp = np.sqrt(d_k)
    tau_solu = np.sqrt(d_k)
    
    # 找到最大激活值，用于统一颜色条范围 (CBAR)
    max_w = 0.0
    for name, kernel_fn in kernel_fns.items():
        if name == "relu":
            w_raw = _retrieve_raw_weights_local(K, q, kernel_fn)
        elif name == "exp":
            w_raw = _retrieve_raw_weights_local(K, q, kernel_fn, tau=tau_exp)
        else: # solu
            w_raw = _retrieve_raw_weights_local(K, q, kernel_fn, tau=tau_solu)
        
        max_w = max(max_w, np.max(w_raw))


    for name, kernel_fn in kernel_fns.items():
        
        if name == "relu":
            w = _retrieve_raw_weights_local(K, q, kernel_fn)
        elif name == "exp":
            # Softmax Attention 标准使用 tau=sqrt(d_k)
            w = _retrieve_raw_weights_local(K, q, kernel_fn, tau=tau_exp)
        else: # solu
            w = _retrieve_raw_weights_local(K, q, kernel_fn, tau=tau_solu)
        
        
        # w = retrieve_raw_weights(K, q, kernel_fn, tau=1.0)
        w = w.reshape(1, -1)  # reshape to [1 x N] for heatmap

        plt.figure(figsize=(10, 1.8))
        sns.heatmap(
            w,
            cmap="viridis",
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            vmin=0,           # 激活值通常非负
            vmax=max_w * 1.05 # 统一颜色范围，方便对比
        )
        plt.title(f"Activation Heatmap — {name.upper()} Kernel (Q=K[0])")
        plt.xlabel(f"Keys (k_1 to k_{N})")
        plt.tight_layout()

        out_path = f"{save_prefix}_{name}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[plot_activation_maps] Saved: {out_path}")


# ----------------------------------------------------
# 主执行块
# ----------------------------------------------------

if __name__ == "__main__":
    # 确保 'plots' 文件夹存在
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    # 定义绘图使用的参数（与实验脚本保持一致）
    d_k_plot = 32
    N_plot = 160 # 选择一个有代表性的 N 值来绘制热图
    
    if not df.empty:
        print("Plotting Accuracy Capacity (High Noise)...")
        plot_accuracy_capacity()
        
        print("Plotting SNR vs N (Linear Y-axis)...")
        plot_snr_vs_n()
        
        print("Plotting Activation Entropy...")
        plot_entropy_vs_n()
        
        # === 新增：绘制 Activation Heatmaps ===
        print("Generating Activation Heatmaps (Correlated Keys)...")
        
        # 1. 生成 Correlated Key
        K_cor = generate_keys_correlated(N_plot, d_k_plot, subspace_dim=6, noise_std=0.1)
        
        # 2. 调用热图函数
        plot_activation_maps(
            K_cor, 
            d_k=d_k_plot,
            q=K_cor[0], # 使用第一个键作为查询
            save_prefix=f"plots/activation_map_cor_N{N_plot}"
        )

    else:
        print("Skipping plotting due to empty or missing data.")