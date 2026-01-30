from coral_patterns.config import BASELINE_DEFAULTS
from coral_patterns.baseline_dla import plot_data_collapse

# gebruik jouw simulate_dla en estimate_fractal_dimension uit dit bestand
# (als dit in hetzelfde bestand staat, hoef je niets te importeren)



def main():
    masses = [250 * 2**i for i in range(5)]

    cfg = BASELINE_DEFAULTS.copy()
    cfg["rng_seed"] = 40   # basis seed (dan seeds 40..49)
    cfg["target_mass"] = 8000  # wordt per N overschreven

    plot_data_collapse(masses, num_seeds=10)
    # plot_fractal_dimension_avg(masses, cfg_base=cfg, num_seeds=10)

if __name__ == "__main__":
    main()
