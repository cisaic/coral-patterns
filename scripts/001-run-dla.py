from coral_patterns.config import DEFAULTS, PLOT_DEFAULTS, MULTIFRACTALITY_DEFAULTS
from coral_patterns.simulation import simulate_dla
from coral_patterns.plot_dla import plot_cluster, plot_mass_radius, plot_mass_over_time, plot_multifractality, plot_growth_probability
from coral_patterns.helpers import estimate_fractal_dimension
from coral_patterns.multifractality import compute_multifractality


def main():
    cfg = DEFAULTS.copy()
    cluster, cluster_history, origin, mass_history, max_r2 = simulate_dla(**cfg)

    plot_cluster(cluster_history, title=PLOT_DEFAULTS["title_cluster"], point_size=PLOT_DEFAULTS["point_size"])
    # plot_mass_radius(cluster, origin, max_r2, title=PLOT_DEFAULTS["title_mass_radius"])
    # plot_mass_over_time(mass_history, title=PLOT_DEFAULTS["title_mass_time"])

    q_vals, sigma_q, growth_probabilities, paths = compute_multifractality(cluster, max_r2, num_walkers=MULTIFRACTALITY_DEFAULTS["num_walkers"], q_range=PLOT_DEFAULTS["q_range"], q_steps=PLOT_DEFAULTS["q_steps"])
    plot_multifractality(q_vals, sigma_q, title=PLOT_DEFAULTS["title_multifractality"])
    fractal_dimension = estimate_fractal_dimension(cluster, origin, max_r2=max_r2)
    print(f"Fractal dimension: {fractal_dimension}")
    plot_growth_probability(growth_probabilities, paths)


if __name__ == "__main__":
    main()