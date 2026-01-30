# 003_plot_overlap_collapse.py
from coral_patterns.data_collapse import plot_data_collapse


def main():
    masses = [250 * 2**i for i in range(5)]
    plot_data_collapse(masses)
    # plot_fractal_dimension(masses)


if __name__ == "__main__":
    main() 
