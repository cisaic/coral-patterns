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

    if not os.path.exists(f"data/multifractality_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}_numwalkers-{MULTIFRACTALITY_DEFAULTS['num_walkers']}.pkl"):
        multifractality_data = compute_multifractality(
            cluster,
            max_r2,
            num_walkers=MULTIFRACTALITY_DEFAULTS["num_walkers"],
            q_range=PLOT_DEFAULTS["q_range"],
            q_steps=PLOT_DEFAULTS["q_steps"],
        )
        
        q_vals = multifractality_data["q_vals"]
        sigma_q = multifractality_data["sigma_q"]
        growth_probabilities = multifractality_data["growth_probabilities"]
        sample_path = multifractality_data["sample_path"]
        slope_at_1 = multifractality_data["slope_at_1"]
        sigma_at_1 = multifractality_data["sigma_at_1"]
        q_tangent = multifractality_data["q_tangent"]
        sigma_tangent = multifractality_data["sigma_tangent"]
        slope_inf = multifractality_data["slope_inf"]
        intercept_inf = multifractality_data["intercept_inf"]
        sigma_at_3 = multifractality_data["sigma_at_3"]
        fractal_dimension = multifractality_data["fractal_dimension"]
    else:
        with open(f"data/multifractality_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}_numwalkers-{MULTIFRACTALITY_DEFAULTS['num_walkers']}.pkl", "rb") as f:
            (
                q_vals,
                sigma_q,
                growth_probabilities,
                sample_path,
                slope_at_1,
                sigma_at_1,
                q_tangent,
                sigma_tangent,
                slope_inf,
                intercept_inf,
                sigma_at_3,
                fractal_dimension
            ) = pickle.load(f)

    plot_multifractality(q_vals, sigma_q, title=PLOT_DEFAULTS["title_multifractality"], cfg=cfg, num_walkers=MULTIFRACTALITY_DEFAULTS["num_walkers"])

    plot_growth_probability(growth_probabilities, cfg=cfg, sample_path=sample_path, num_walkers=MULTIFRACTALITY_DEFAULTS["num_walkers"])


if __name__ == "__main__":
    main()