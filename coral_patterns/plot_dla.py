import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Set, Dict

from .helpers import estimate_fractal_dimension


def plot_cluster(
    cluster_history: List[Tuple[int, int]],
    title: str = "",
    point_size: float = 0.6
) -> None:
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


def plot_mass_radius(cluster, origin, max_r2, title):
    """
    Log-log plot of M(r) vs r + fitted power law line.
    """
    # read the results from the fractal dimension estimation.
    out = estimate_fractal_dimension(cluster, origin, max_r2=max_r2)
    D = out["D"]
    c = out["intercept"]
    r_list = out["r_list"]
    M_list = out["M_list"]

    # fitted curve: M_fit = exp(c) * r^D
    M_fit = [math.exp(c) * (r ** D) for r in r_list]

    # loglog plot generation 
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


def plot_mass_over_time(mass_history, title = "Cluster mass over time"):
    """M(t) plot."""
    plt.figure(figsize=(7, 4))
    plt.plot(mass_history)
    plt.xlabel("growth step")
    plt.ylabel("M(t)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

def plot_multifractality(q_range, sigma_q, title: str = "Multifractality"):
    """
    Plot multifractality.
    """
    # measure slope of the curve
    coefficients = np.polyfit(q_range, sigma_q, deg=1)
    slope = coefficients[0]
    intercept = coefficients[1]

    print(f"Slope (scaling exponent): {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")


    plt.figure(figsize=(8, 6))
    plt.plot(q_range, slope * q_range + intercept, 'r-', linewidth=2, label=f'Linear fit: σ(q) = {slope:.3f}q + {intercept:.3f}')
    plt.plot(q_range, sigma_q, 'o', label='Data', markersize=2)
    plt.xlabel('q')
    plt.ylabel('σ(q)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_growth_probability(
    growth_probabilities: Dict[Tuple[int, int], float],
    paths: List[List[Tuple[int, int]]],
    title: str = ""
) -> None:
    """Plot growth probability."""

    xs = [x for (x, _) in growth_probabilities.keys()]
    ys = [y for (_, y) in growth_probabilities.keys()]
    colors = list(growth_probabilities.values())

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(xs, ys, s=0.6, c=colors, cmap='cool', vmin=min(colors), vmax=max(colors))
    for path in paths[:1]:
        if path:
            # Plot the starting point as a red dot
            plt.plot(path[0][0], path[0][1], 'ro', markersize=2)
            # Plot the rest of the path as a grey line
            if len(path) > 1:
                plt.plot([x for (x, _) in path], [y for (_, y) in path], color='grey', alpha=0.1, linewidth=0.6)
    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    plt.title("Growth Probabilities")
    plt.colorbar(scatter, label='Growth Probability')
    plt.show()