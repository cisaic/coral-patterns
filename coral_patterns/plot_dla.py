import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Set, Dict, Any
from scipy.interpolate import interp1d

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

    # Interpolate sigma_q to a finer grid
    f = interp1d(q_range, sigma_q, kind='cubic', fill_value='extrapolate')
    
    # Measure slope as q -> infinity
    q_inf = np.linspace(np.max(q_range) - 10, np.max(q_range), 10)
    sigma_q_inf = f(q_inf)  # Interpolate values
    slope_inf, intercept_inf = np.polyfit(q_inf, sigma_q_inf, deg=1)

    print(f"Slope at high q: {slope_inf:.4f}")

    plt.figure(figsize=(8, 6))

    q_range = np.array(q_range)
    sigma_q = np.array(sigma_q)
    
    # Fit the entire curve to get derivative
    f = interp1d(q_range, sigma_q, kind='cubic', fill_value='extrapolate')
    
    # Get the slope at q=1 using numerical derivative
    dq = 0.001
    slope_at_1 = (f(1 + dq) - f(1 - dq)) / (2 * dq)
    sigma_at_1 = f(1)
    
    # Tangent line: y = m(x - x0) + y0
    # y = slope_at_1 * (q - 1) + sigma_at_1
    q_tangent = np.linspace(-2, 5, 100)
    sigma_tangent = slope_at_1 * (q_tangent - 1) + sigma_at_1
    
    print(f"Slope at q=1: {slope_at_1:.4f}")

    # plot data
    plt.plot(q_range, sigma_q, 'o', label='Data', color='darkgrey', markersize=2)
    
    # Plot tangent line
    plt.plot(q_tangent, sigma_tangent, 'r-', linewidth=2, color='magenta',
             label=f'Tangent at q=1: σ(q) = {slope_at_1:.3f}(q-1) + {sigma_at_1:.3f}')
    # plot slope as q -> infinity
    plt.plot(q_inf, slope_inf * q_inf + intercept_inf, 'r-', linewidth=2, color='teal', label=f'Linear fit: σ(q) = {slope_inf:.3f}q + {intercept_inf:.3f}')

    # plot a single point at q=3
    # Find the index of the value in q_range closest to 3
    idx_closest_to_3 = np.abs(np.array(q_range) - 3).argmin()
    plt.plot(q_range[idx_closest_to_3], sigma_q[idx_closest_to_3], 'o', color='blue', label=f'q≈3: σ(q) = {sigma_q[idx_closest_to_3]:.3f}', markersize=4)

    # plot a single point at q=1
    idx_closest_to_1 = np.abs(np.array(q_range) - 1).argmin()
    plt.plot(q_range[idx_closest_to_1], sigma_q[idx_closest_to_1], 'o', color='magenta', label=f'q≈1:', markersize=4)

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
    """Plot growth probability."""

    xs = [x for (x, _) in growth_probabilities.keys()]
    ys = [y for (_, y) in growth_probabilities.keys()]
    colors = list(growth_probabilities.values())

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(xs, ys, s=0.6, c=colors, cmap='cool', vmin=min(colors), vmax=max(colors))
    
    plt.plot(sample_path[0][0], sample_path[0][1], 'ro', markersize=2)
    # Plot the rest of the path as a grey line
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