import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("kernel_capacity_results.csv")

def plot_accuracy():
    sns.set(style="whitegrid")
    for key_mode in ["indep", "correlated"]:
        plt.figure(figsize=(6,4))
        sub = df[df["key_mode"]==key_mode]

        for kernel in ["relu", "exp", "solu"]:
            temp = sub[sub["kernel"]==kernel]
            plt.errorbar(
                temp["N"], temp["acc_mean"],
                yerr=temp["acc_std"],
                label=kernel
            )

        plt.title(f"Retrieval Accuracy vs N [{key_mode}]")
        plt.xlabel("N (memory load)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"plots/acc_{key_mode}.png")
        plt.show()

def plot_snr():
    sns.set(style="whitegrid")
    for key_mode in ["indep", "correlated"]:
        plt.figure(figsize=(6,4))
        sub = df[df["key_mode"]==key_mode]

        for kernel in ["relu", "exp", "solu"]:
            temp = sub[sub["kernel"]==kernel]
            plt.errorbar(
                temp["N"], temp["snr_mean"],
                yerr=temp["snr_std"],
                label=kernel
            )

        plt.title(f"SNR vs N [{key_mode}]")
        plt.xlabel("N (memory load)")
        plt.ylabel("SNR")
        plt.legend()
        plt.savefig(f"plots/snr_{key_mode}.png")
        plt.show()

if __name__ == "__main__":
    plot_accuracy()
    plot_snr()

