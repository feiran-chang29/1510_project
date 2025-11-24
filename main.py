# main.py

from run_experiments import run_all
from visualize import generate_all_plots

def main():
    print("\n=== Running Associative Memory Experiments ===")
    
    # 1. Run experiments and output CSV
    df = run_all()

    print("\n=== Generating Visualizations ===")
    generate_all_plots("kernel_capacity_results.csv")

    print("\nAll experiments and plots complete!")
    print("Check the 'results/' and 'plots/' folders.")

if __name__ == "__main__":
    main()
