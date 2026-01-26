import math
import matplotlib.pyplot as plt
import numpy as np

from coral_patterns.dla_baseline import estimate_fractal_dimension


def plot_cluster(cluster_history, title="", point_size=0.6):
    """Scatter plot of occupied lattice sites."""
    xs = [x for (x, _) in cluster_history]
    ys = [y for (_, y) in cluster_history]

    # create a colour map for the history of the cluster growth, so the older sites are different colours from the younger ones
    history = np.linspace(0, 1, len(cluster_history))

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=point_size, c=history, cmap='cool', vmin=0, vmax=1)
    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def plot_mass_radius(cluster, origin, max_r2, title=""):
    """Log-log plot of M(r) vs r + fitted power law line."""
    out = estimate_fractal_dimension(cluster, origin, max_r2=max_r2)
    D = out["D"]
    c = out["intercept"]
    r_list = out["r_list"]
    M_list = out["M_list"]

    # fitted curve: M_fit = exp(c) * r^D
    M_fit = [math.exp(c) * (r ** D) for r in r_list]

    plt.figure(figsize=(6, 5))
    plt.loglog(r_list, M_list, marker="o", linestyle="None", label="M(r)")
    plt.loglog(r_list, M_fit, linestyle="-", label=f"fit slope D ≈ {D:.3f}")
    plt.xlabel("r")
    plt.ylabel("M(r)")
    plt.grid(True, which="both", alpha=0.25)

    if title:
        plt.title(title)
    else:
        plt.title(f"Mass-radius scaling (fit window [{out['r_lo']:.1f}, {out['r_hi']:.1f}])")

    plt.legend()
    plt.show()


def plot_mass_over_time(mass_history, title="Cluster mass over time"):
    """M(t) plot."""
    plt.figure(figsize=(7, 4))
    plt.plot(mass_history)
    plt.xlabel("growth step")
    plt.ylabel("M(t)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()
