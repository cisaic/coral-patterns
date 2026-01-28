import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Set, Dict, Any
from .multifractality import multifractality_fit

from .helpers import estimate_fractal_dimension


def plot_cluster(
    cluster_history: List[Tuple[int, int]],
    cfg: Dict[str, Any],
    title: str = "",
    point_size: float = 0.6,
    show_fig: bool = False
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
    if show_fig:
        plt.show()
    else:
        plt.savefig(f"plots/cluster_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}.png")
        plt.close()

def plot_mass_radius(
    cluster: Set[Tuple[int, int]],
    origin: Tuple[int, int],
    max_r2: float,
    cfg: Dict[str, Any],
    title: str = "",
    show_fig: bool = False
) -> None:
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
    if show_fig:
        plt.show()
    else:
        plt.savefig(f"plots/mass_radius_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}.png")
        plt.close()


def plot_mass_over_time(
    mass_history: List[float],
    cfg: Dict[str, Any],
    title: str = "Cluster mass over time",
    show_fig: bool = False
) -> None:
    """M(t) plot."""
    plt.figure(figsize=(7, 4))
    plt.plot(mass_history)
    plt.xlabel("growth step")
    plt.ylabel("M(t)")
    plt.title(title)
    plt.grid(alpha=0.3)
    if show_fig:
        plt.show()
    else:
        plt.savefig(f"plots/mass_over_time_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}.png")
        plt.close()


def plot_multifractality(
    q_range: List[float],
    sigma_q: List[float],
    cfg: Dict[str, Any],
    num_walkers: int,
    title: str = "Multifractality",
    show_fig: bool = False
) -> None:
    """Plot multifractality."""

    multifractality_data = multifractality_fit(q_range, sigma_q)
    slope_at_1 = multifractality_data["slope_at_1"]
    sigma_at_1 = multifractality_data["sigma_at_1"]
    q_tangent = multifractality_data["q_tangent"]
    sigma_tangent = multifractality_data["sigma_tangent"]
    slope_inf = multifractality_data["slope_inf"]
    intercept_inf = multifractality_data["intercept_inf"]
    sigma_at_3 = multifractality_data["sigma_at_3"]

    plt.figure(figsize=(8, 6))

    # plot data
    plt.plot(q_range, sigma_q, 'o', label='Data', color='darkgrey', markersize=2)

    # plot a single point at q=3
    plt.plot(3, sigma_at_3, 'o', color='blue', label=f'q≈3: σ(q) = {sigma_at_3:.3f}', markersize=4)

    # plot a single point at q=1
    plt.plot(1, sigma_at_1, 'o', color='magenta', label=f'q≈1:', markersize=4)
    
    # Plot tangent line at q=1
    plt.plot(q_tangent, sigma_tangent, '-', linewidth=2, color='magenta', 
         label=f'Tangent at q≈1: σ(q) = {slope_at_1:.3f}(q-1) + {sigma_at_1:.3f}')

    # plot slope as q -> infinity
    plt.plot(q_range[-10:], slope_inf * q_range[-10:] + intercept_inf, '-', linewidth=2, color='teal', 
         label=f'Linear fit: σ(q) = {slope_inf:.3f}q + {intercept_inf:.3f}')
         
    plt.xlabel('q')
    plt.ylabel('σ(q)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if show_fig:
        plt.show()
    else:
        plt.savefig(f"plots/multifractality_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}_numwalkers-{num_walkers}.png")
        plt.close()


def plot_growth_probability(
    growth_probabilities: Dict[Tuple[int, int], float],
    cfg: Dict[str, Any],
    sample_path: List[Tuple[int, int]],
    num_walkers: int,
    title: str = "",
    show_fig: bool = False
) -> None:
    """Plot growth probability with an example path taken by a walker."""

    xs = [x for (x, _) in growth_probabilities.keys()]
    ys = [y for (_, y) in growth_probabilities.keys()]
    colors = list(growth_probabilities.values())

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(xs, ys, s=0.6, c=colors, cmap='cool', vmin=min(colors), vmax=max(colors))
    
    # Plot the first point of the sample path as a red dot
    plt.plot(sample_path[0][0], sample_path[0][1], 'ro', markersize=2)
    
    # Plot the rest of the walker's path as a grey line
    if len(sample_path) > 1:
        plt.plot([x for (x, _) in sample_path], [y for (_, y) in sample_path], color='grey', alpha=0.3, linewidth=0.6)

    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    plt.title("Growth Probabilities")
    plt.colorbar(scatter, label='Growth Probability')
    if show_fig:
        plt.show()
    else:
        plt.savefig(f"plots/growth_probabilities_mass-{cfg['target_mass']}_gm-{cfg['growth_mode']}_f-{cfg['friendliness']}_seed-{cfg['rng_seed']}_numwalkers-{num_walkers}.png")
        plt.close()