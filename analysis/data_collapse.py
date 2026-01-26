import math
import matplotlib.pyplot as plt

from coral_patterns.config import DEFAULTS
from coral_patterns.Simulation import simulate_dla
from coral_patterns.helpers import estimate_fractal_dimension


import math
import matplotlib.pyplot as plt

from coral_patterns.config import DEFAULTS
from coral_patterns.Simulation import simulate_dla
from coral_patterns.helpers import estimate_fractal_dimension


def compute_collapse_curve(N, cfg_base, seed_offset=0):
    """
    Run 1 DLA simulatie en return (x, y, D) voor de collapse-plot.
    """

    # copy the configurations
    cfg = cfg_base.copy()
    cfg["target_mass"] = int(N)
    cfg["rng_seed"] = cfg_base.get("rng_seed", 1) + seed_offset

    cluster, _, origin, _, max_r2 = simulate_dla(**cfg)

    out = estimate_fractal_dimension(cluster, origin, max_r2=max_r2)
    r_list = out["r_list"]
    M_list = out["M_list"]

    # fractal dimensie
    D = out.get("D", out.get("slope"))
    if D is None:
        raise KeyError("Missing 'D'/'slope' in estimate_fractal_dimension output")

    # normalize radius
    R = math.sqrt(max_r2) if max_r2 and max_r2 > 0 else 1.0

    # collapse coordinates
    x = [(r / R) ** D for r in r_list]
    y = [M / N for M in M_list]

    return x, y, D


def plot_data_collapse(masses, seed=None):
    """Plot collapse curves for a list of masses."""
    cfg_base = DEFAULTS.copy()
    if seed is not None:
        cfg_base["rng_seed"] = seed

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    # iterate over the target masses
    for i, N in enumerate(masses):
        x, y, D = compute_collapse_curve(N, cfg_base, seed_offset=i)

        # initialize loglog plot
        ax.loglog(
            x, y,
            "o",
            markersize=4,
            alpha=0.7,
            label=f"N={N}, D={D:.3f}"
        )

    # graph styling
    ax.set_xlabel(r"$(r / R)^D$")
    ax.set_ylabel(r"$M(r) / N$")
    ax.set_title("Fractal data collapse")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    return fig, ax
