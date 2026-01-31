import math
import matplotlib.pyplot as plt
import numpy as np

from coral_patterns.config import DEFAULTS
from coral_patterns.simulation import simulate_dla
from coral_patterns.helpers import estimate_fractal_dimension


def average_over_seeds(N, cfg_base, seeds):
    """
    Run multiple seeds for mass N and return mean curves + 95% CI (for the mean).
    Returns:
    dr_mean, M_mean, M_lo, M_hi, D_mean, R_mean
    """
    r_lists = []
    M_lists = []
    Ds = []
    Rs = []

    for s in seeds:
        cfg = cfg_base.copy()
        cfg["target_mass"] = int(N)
        cfg["rng_seed"] = cfg_base.get("rng_seed", 1) + int(s)

        cluster, _, origin, _, max_r2 = simulate_dla(**cfg)
        out = estimate_fractal_dimension(cluster, origin, max_r2=max_r2)

        r_lists.append(np.asarray(out["r_list"], dtype=float))
        M_lists.append(np.asarray(out["M_list"], dtype=float))
        Ds.append(out.get("D", out.get("slope")))
        Rs.append(math.sqrt(max_r2) if max_r2 and max_r2 > 0 else 1.0)

    # assume r grids are identical
    r_arr = np.vstack(r_lists)          # (n_seeds, n_points)
    M_arr = np.vstack(M_lists)          # (n_seeds, n_points)

    r_mean = r_arr.mean(axis=0)
    M_mean = M_arr.mean(axis=0)

    n = M_arr.shape[0]
    if n > 1:
        M_std = M_arr.std(axis=0, ddof=1)
        se = M_std / math.sqrt(n)

        # 95% CI for mean: normal approx 
        z = 1.96
        M_lo = M_mean - z * se
        M_hi = M_mean + z * se
    else:
        M_lo = M_mean.copy()
        M_hi = M_mean.copy()

    D_mean = float(np.mean(Ds))
    R_mean = float(np.mean(Rs))

    return r_mean, M_mean, M_lo, M_hi, D_mean, R_mean


def run_dla_and_measure(N, cfg_base, seed_offset=0):
    """
    Simulate DLA for target mass N and return (r_list, M_list, D, max_r2).
    """
    cfg = cfg_base.copy()
    cfg["target_mass"] = int(N)
    cfg["rng_seed"] = cfg_base.get("rng_seed", 1) + seed_offset

    cluster, _, origin, _, max_r2 = simulate_dla(**cfg)

    out = estimate_fractal_dimension(cluster, origin, max_r2=max_r2)
    r_list = out["r_list"]
    M_list = out["M_list"]

    D = out.get("D", out.get("slope"))
    if D is None:
        raise KeyError("Missing 'D'/'slope' in estimate_fractal_dimension output")

    return r_list, M_list, D, max_r2


def plot_data_collapse(masses, num_seeds=10):
    cfg_base = DEFAULTS.copy()
    seeds = list(range(num_seeds))

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    for N in masses:
        r, M, M_lo, M_hi, D, R = average_over_seeds(N, cfg_base, seeds)

        # construct axes
        x = (r / R) ** D
        y = M / N

        # construct confidence intervals
        y_lo = M_lo / N
        y_hi = M_hi / N

        # only datapoints are plotted
        ax.loglog(x, y, "o", markersize=4, alpha=0.8, label=f"N={N}, ⟨D⟩={D:.3f}")

        # 95% CI band
        ax.fill_between(x, y_lo, y_hi, alpha=0.20)

    ax.set_xlabel(r"$(r / R)^D$", fontsize=18)
    ax.set_ylabel(r"$M(r) / N$", fontsize=18)
    ax.set_title(f"Mass-radius scaling 5-Moore neighborhood", fontsize=18)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
    return fig, ax


