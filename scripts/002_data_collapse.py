# 003_plot_overlap_collapse.py
from analysis.data_collapse import plot_data_collapse


def main():
    masses = [250 * 2**i for i in range(5)] + [8000, 10000]
    plot_data_collapse(masses)


if __name__ == "__main__":
    main() 
