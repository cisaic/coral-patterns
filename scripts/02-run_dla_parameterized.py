from coral_patterns.dla_parameterized import DEFAULTS, simulate_dla
from coral_patterns.plotting_dla import plot_cluster, plot_mass_radius, plot_mass_over_time


PLOT_DEFAULTS = {
    "point_size": 0.6,
    "title_cluster": "Parameterized DLA (8-neighborhood)",
    "title_mass_radius": "Mass-radius scaling (log-log)",
    "title_mass_time": "Baseline growth: M(t)",
}


def main():
    cfg = DEFAULTS.copy()
    cluster, cluster_history, origin, mass_history, max_r2 = simulate_dla(**cfg)

    plot_cluster(cluster_history, title=PLOT_DEFAULTS["title_cluster"], point_size=PLOT_DEFAULTS["point_size"])
    plot_mass_radius(cluster, origin, max_r2, title=PLOT_DEFAULTS["title_mass_radius"])
    plot_mass_over_time(mass_history, title=PLOT_DEFAULTS["title_mass_time"])


if __name__ == "__main__":
    main()
