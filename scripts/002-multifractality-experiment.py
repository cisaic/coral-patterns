from coral_patterns.config import DEFAULTS, PLOT_DEFAULTS, MULTIFRACTALITY_DEFAULTS
from coral_patterns.simulation import simulate_dla
from coral_patterns.plot_dla import plot_cluster, plot_mass_radius, plot_mass_over_time, plot_multifractality, plot_growth_probability
from coral_patterns.helpers import estimate_fractal_dimension
from coral_patterns.multifractality import compute_multifractality
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def run_single_multifractality_experiment(cluster, cluster_history, max_r2, cfg):
    plot_cluster(cluster_history, cfg=cfg, title=f"Cluster of size {len(cluster)} with growth mode {cfg['growth_mode']} and friendliness {cfg['friendliness']} seed {cfg['rng_seed']}", point_size=PLOT_DEFAULTS["point_size"])

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

def plot_heatmap(data, title, xlabel, ylabel):
    """Plot a heatmap of the data."""

    friendliness_values = [0, 0.25, 0.5, 0.75, 1]
    growth_mode_values = [-1, -0.5, 0, 0.5, 1]

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(growth_mode_values)), labels=growth_mode_values)
    ax.set_yticks(range(len(friendliness_values)), labels=friendliness_values)

    # Loop over data dimensions and create text annotations.
    for i in range(len(friendliness_values)):
        for j in range(len(growth_mode_values)):
            text = ax.text(j, i, data[i, j],
                        ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(f"plots/heatmap_{title}.png")
    plt.close()

def plot_multi_heatmap(target_data, result_data, title, xlabel, ylabel):
    """Plot heatmap with two values per square: target (upper triangle) vs result (lower triangle)"""

    friendliness_values = [0, 0.25, 0.5, 0.75, 1]
    growth_mode_values = [-1, -0.5, 0, 0.5, 1]

    fig, ax = plt.subplots()
    rows, cols = target_data.shape

    cmap = plt.get_cmap('viridis')

    vmin = min(np.min(target_data), np.min(result_data))
    vmax = max(np.max(target_data), np.max(result_data))

    # Create masks for upper and lower triangles
    rows, cols = target_data.shape
    mask_upper = np.tril(np.ones((rows, cols), dtype=bool))
    mask_lower = np.triu(np.ones((rows, cols), dtype=bool))
    
    # Plot reference (upper triangle)
    ref_masked = np.ma.masked_where(mask_upper, target_data)
    im1 = ax.imshow(ref_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Plot experimental (lower triangle)
    exp_masked = np.ma.masked_where(mask_lower, result_data)
    im2 = ax.imshow(exp_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Set labels
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(growth_mode_values)
    ax.set_yticklabels(friendliness_values)

    # Add grid
    ax.set_xticks(np.arange(cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    # add axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Colorbar
    cbar = plt.colorbar(im1, ax=ax)
    cbar.set_label('Value')
    
    # set title
    fig.suptitle(title)

    # set subtitle
    ax.set_title('Split Heatmap: Upper=Target, Lower=Experiment result')
    plt.tight_layout()
    plt.show()

def run_multifractality_experiment(cluster, cluster_history, max_r2, cfg, runs=10):
    """
    Run the multifractality experiment for each combination of friendliness and growth mode.
    """
    # for each combination of friendliness and growth mode, run the experiment 10 times (10 seeds)
    # plot the average of the 10 runs for each 

    friendliness_values = [0, 0.25, 0.5, 0.75, 1]
    growth_mode_values = [-1, -0.5, 0, 0.5, 1]
    slope_at_1_values = np.zeros((len(friendliness_values), len(growth_mode_values))) 
    slope_inf_values = np.zeros((len(friendliness_values), len(growth_mode_values)))
    sigma_at_3_values = np.zeros((len(friendliness_values), len(growth_mode_values)))
    fractal_dimension_values = np.zeros((len(friendliness_values), len(growth_mode_values)))

    for i, friendliness in enumerate(friendliness_values):
        for j, growth_mode in enumerate(growth_mode_values):
            for seed in range(runs):
                print(f"Running experiment for friendliness: {friendliness} - growth mode: {growth_mode} - seed: {seed}")
                cfg["friendliness"] = friendliness
                cfg["growth_mode"] = growth_mode
                cfg["rng_seed"] = seed
                multifractality_data = compute_multifractality(cluster, max_r2, num_walkers=MULTIFRACTALITY_DEFAULTS["num_walkers"], q_range=PLOT_DEFAULTS["q_range"], q_steps=PLOT_DEFAULTS["q_steps"])

                sigma_q = multifractality_data["sigma_q"]
                growth_probabilities = multifractality_data["growth_probabilities"]
                sample_path = multifractality_data["sample_path"]

                q_vals = multifractality_data["q_vals"]
                slope_at_1 = multifractality_data["slope_at_1"]
                slope_inf = multifractality_data["slope_inf"]
                sigma_at_3 = multifractality_data["sigma_at_3"]
                fractal_dimension = multifractality_data["fractal_dimension"]

                slope_at_1_values[i, j] += slope_at_1
                slope_inf_values[i, j] += slope_inf
                sigma_at_3_values[i, j] += sigma_at_3
                fractal_dimension_values[i, j] += fractal_dimension["D"]

                if seed == 0:
                    plot_cluster(cluster_history, cfg=cfg, title=f"Cluster of size {len(cluster)} with growth mode {cfg['growth_mode']} and friendliness {cfg['friendliness']} seed {cfg['rng_seed']}", point_size=PLOT_DEFAULTS["point_size"])
                    plot_multifractality(q_vals, sigma_q, title=f"Multifractality - Friendliness: {friendliness} - Growth Mode: {growth_mode} - Mass: {len(cluster)}", cfg=cfg, num_walkers=MULTIFRACTALITY_DEFAULTS["num_walkers"])
                    plot_growth_probability(growth_probabilities, cfg=cfg, sample_path=sample_path, num_walkers=MULTIFRACTALITY_DEFAULTS["num_walkers"], title=f"Growth Probability - Friendliness: {friendliness} - Growth Mode: {growth_mode} - Mass: {len(cluster)} - Num Walkers: {MULTIFRACTALITY_DEFAULTS['num_walkers']}")

    slope_at_1_values /= runs
    slope_inf_values /= runs
    sigma_at_3_values /= runs
    fractal_dimension_values /= runs

    inv_d_values = 1 / fractal_dimension_values
    slope_inf_targets = fractal_dimension_values ** 2 - fractal_dimension_values

    # save the values to a npz file
    np.savez(f"data/multifractality_experiment_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_numwalkers-{MULTIFRACTALITY_DEFAULTS['num_walkers']}.npz", slope_at_1_values=slope_at_1_values, slope_inf_values=slope_inf_values, sigma_at_3_values=sigma_at_3_values, fractal_dimension_values=fractal_dimension_values, inv_d_vals=inv_d_vals, q_inf_slope=q_inf_slope)

    return {
        "slope_at_1_values": slope_at_1_values, 
        "slope_inf_values": slope_inf_values, 
        "sigma_at_3_values": sigma_at_3_values, 
        "fractal_dimension_values": fractal_dimension_values, 
        "inv_d_values": inv_d_values, 
        "slope_inf_targets": slope_inf_targets
    }

def main():
    cfg = DEFAULTS.copy()

    if not os.path.exists(f"data/dla_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}.pkl"):
        cluster, cluster_history, origin, mass_history, max_r2 = simulate_dla(**cfg)
    else:
        with open(f"data/dla_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}.pkl", "rb") as f:
            cluster, cluster_history, origin, mass_history, max_r2 = pickle.load(f)

    # run_single_multifractality_experiment(cluster, cluster_history, max_r2, cfg)
    results = run_multifractality_experiment(cluster, cluster_history, max_r2, cfg)

    slope_at_1_values = results["slope_at_1_values"]
    inv_d_values = results["inv_d_values"]

    slope_inf_values = results["slope_inf_values"]
    slope_inf_targets = results["slope_inf_targets"]

    sigma_at_3_values = results["sigma_at_3_values"]
    
    fractal_dimension_values = results["fractal_dimension_values"]

    # plot slope_at_1_values vs inv_d_vals
    plot_multi_heatmap(slope_at_1_values, inv_d_values, "σ(q) Slope at q=1 vs Inverse Fractal Dimension", "Growth Mode (horizontal → vertical)", "Friendliness (branchy → friendly)")

    # plot slope_inf_values vs slope_inf_targets
    plot_multi_heatmap(slope_inf_values, slope_inf_targets, "σ(q) Slope → ∞ vs Target Slope D^2 - D", "Growth Mode (horizontal → vertical)", "Friendliness (branchy → friendly)")

    # plot sigma_at_3_values
    plot_heatmap(np.exp(sigma_at_3_values), "σ(q) at q=3", "Growth Mode (horizontal → vertical)", "Friendliness (branchy → friendly)")

    # plot fractal_dimension_values
    plot_heatmap(fractal_dimension_values, "Fractal Dimension", "Growth Mode (horizontal → vertical)", "Friendliness (branchy → friendly)")



if __name__ == "__main__":
    main()