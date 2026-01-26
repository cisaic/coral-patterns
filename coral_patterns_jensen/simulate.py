# simulation.py
"""
Main simulation loop for the 2D DLA-based coral growth model.

This module orchestrates:
- random walker injection and sticking
- growth-mode filtering (smin, smax)
- optional branching via leaders
- physical time tracking

It combines utilities from helpers.py and run.py.
"""

import random

from config import DEFAULTS, PAPER_PARAMS
from helpers import (
    radius_from_r2,
    sqdist_point,
)
from run import (
    build_frontier,
    update_frontier,
    compute_tips,
    pick_branch_direction,
    preferred_site_from_tip,
    count_occupied_neighbors,
    run_walker,
)


def simulate_dla_with_paper_params(cfg, paper):
    """
    Run a DLA simulation with growth-mode control inspired by the Llabrés et al. paper.
    Model parameters (growth window smin/smax, elongation rate, branching length)
    """
    rng = random.Random(cfg["rng_seed"])
    origin = (0, 0)

    # Initial seed
    cluster = {origin}
    frontier = build_frontier(cluster)

    max_r2 = 0
    mass_history = [1]

    # Physical time bookkeeping
    nu = float(paper["nu_mm_per_year"])
    delta = float(paper["delta_sub_mm"])
    dt_years = (delta / nu) if nu > 0 else 0.0

    time_history_years = [0.0]
    t_years = 0.0

    # Branching parameters (used only if branching is enabled)
    l_br_mm = float(paper["l_br_mm"])
    l_br_steps = max(1, int(round(l_br_mm / max(delta, 1e-9))))
    theta_deg = float(paper["theta_deg"])

    # Growth-mode window
    smin = float(paper["smin"])
    smax = float(paper["smax"])

    leaders = []
    preferred_targets = set()

    def refresh_preferred_targets():
        """
        Update the set of preferred frontier sites based on active leaders.
        """
        preferred_targets.clear()
        for L in leaders:
            tip = L["tip"]
            target = preferred_site_from_tip(tip, L["dir"])
            if target in frontier:
                preferred_targets.add(target)


    while len(cluster) < cfg["target_mass"]:
        # Update launch and kill radii based on current size
        R = radius_from_r2(max_r2)
        launch_r = R + cfg["launch_margin"]
        kill_r = launch_r + cfg["kill_margin"]

        # Possibly introduce a new branch leader
        if (len(cluster) > 1) and (len(cluster) % l_br_steps == 0):
            tips = compute_tips(cluster)
            if tips:
                tip = tips[rng.randrange(len(tips))]
                leaders.append({
                    "tip": tip,
                    "dir": pick_branch_direction(theta_deg, rng)
                })
                refresh_preferred_targets()

        # Run a single random walker
        stuck = run_walker(
            cluster=cluster,
            frontier=frontier,
            origin=origin,
            launch_radius=launch_r,
            kill_radius=kill_r,
            max_steps=cfg["max_steps_per_walker"],
            rng=rng,
            smin=smin,
            smax=smax,
            preferred_targets=preferred_targets,
        )

        if stuck is None:
            continue

        # Attach particle
        cluster.add(stuck)
        update_frontier(stuck, cluster, frontier)

        # Update leader tips if growth continues from a new site
        for L in leaders:
            if (
                count_occupied_neighbors(L["tip"], cluster) != 1
                and count_occupied_neighbors(stuck, cluster) == 1
            ):
                L["tip"] = stuck

        refresh_preferred_targets()

        # Update size and time
        r2 = sqdist_point(stuck, origin)
        if r2 > max_r2:
            max_r2 = r2

        mass_history.append(len(cluster))
        t_years += dt_years
        time_history_years.append(t_years)

        if cfg["log_every"] and (len(cluster) % cfg["log_every"] == 0):
            print(
                f"[DLA+paper] mass={len(cluster)}  "
                f"R~{radius_from_r2(max_r2)}  "
                f"t~{t_years:.2f} yr"
            )

    return cluster, origin, mass_history, time_history_years, max_r2


# Main functionality
if __name__ == "__main__":
    from plot import (
        plot_cluster,
        plot_mass_vs_time,
        plot_mass_radius,
    )

    cfg = DEFAULTS.copy()

    cluster, origin, mass_hist, time_hist_years, max_r2 = simulate_dla_with_paper_params(
        cfg, PAPER_PARAMS
    )

    print(
        "Done. Final mass:",
        len(cluster),
        "Final time (years):",
        time_hist_years[-1],
    )

    plot_cluster(cluster, title="DLA + paper-like controls (2D)", point_size=0.6)
    plot_mass_vs_time(mass_hist, time_hist_years, title="Mass vs physical time (years)")
    plot_mass_radius(cluster, origin, max_r2, title="Mass-radius scaling (log-log)")
