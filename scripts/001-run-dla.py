from coral_patterns.config import DEFAULTS, PLOT_DEFAULTS, MULTIFRACTALITY_DEFAULTS
from coral_patterns.simulation import simulate_dla
from coral_patterns.plot_dla import plot_cluster, plot_mass_radius, plot_mass_over_time, plot_multifractality, plot_growth_probability
from coral_patterns.helpers import estimate_fractal_dimension
from coral_patterns.multifractality import compute_multifractality
import pickle
import os


def main():
    cfg = DEFAULTS.copy()

    if not os.path.exists(f"data/dla_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}.pkl"):
        cluster, cluster_history, origin, mass_history, max_r2 = simulate_dla(**cfg)
    else:
        with open(f"data/dla_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}.pkl", "rb") as f:
            cluster, cluster_history, origin, mass_history, max_r2 = pickle.load(f)

    plot_cluster(cluster_history, cfg=cfg, title=PLOT_DEFAULTS["title_cluster"], point_size=PLOT_DEFAULTS["point_size"])
    # plot_mass_radius(cluster, origin, max_r2, cfg=cfg title=PLOT_DEFAULTS["title_mass_radius"])
    # plot_mass_over_time(mass_history, cfg=cfg, title=PLOT_DEFAULTS["title_mass_time"])

    if not os.path.exists(f"data/multifractality_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}_numwalkers-{MULTIFRACTALITY_DEFAULTS['num_walkers']}.pkl"):
        q_vals, sigma_q, growth_probabilities, sample_path = compute_multifractality(cluster, max_r2, num_walkers=MULTIFRACTALITY_DEFAULTS["num_walkers"], q_range=PLOT_DEFAULTS["q_range"], q_steps=PLOT_DEFAULTS["q_steps"])
    else:
        with open(f"data/multifractality_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}_numwalkers-{MULTIFRACTALITY_DEFAULTS['num_walkers']}.pkl", "rb") as f:
            q_vals, sigma_q, growth_probabilities, sample_path = pickle.load(f)

    plot_multifractality(q_vals, sigma_q, title=PLOT_DEFAULTS["title_multifractality"], cfg=cfg, num_walkers=MULTIFRACTALITY_DEFAULTS["num_walkers"])
    
    fractal_dimension = estimate_fractal_dimension(cluster, origin, max_r2=max_r2)
    print(f"Fractal dimension: {fractal_dimension}")
    
    plot_growth_probability(growth_probabilities, cfg=cfg, sample_path=sample_path, num_walkers=MULTIFRACTALITY_DEFAULTS["num_walkers"])


if __name__ == "__main__":
    main()