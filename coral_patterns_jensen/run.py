# growth.py
import math

from config import NEIGHBORS8
from helpers import (
    in_domain,
    sqdist,
    xi_verticalness,
    random_step,
    launch_point,
)

def build_frontier(cluster):
    """
    Frontier = all empty sites adjacent (4-neighborhood) to the occupied cluster.
    If a walker lands on a frontier site, it can stick. We keep this updated throughout the simulation with the next funtion
    """
    frontier = set()
    for (x, y) in cluster:
        for dx, dy in NEIGHBORS8:
            q = (x + dx, y + dy)
            if q not in cluster and in_domain(q[0], q[1]):
                frontier.add(q)
    return frontier

def update_frontier(new_site, cluster, frontier):
    """
    We just attached at new_site:
    - new_site becomes occupied -> remove it from frontier if it was there.
    - its empty neighbors become frontier sites -> add them.
    """
    frontier.discard(new_site)
    x, y = new_site
    for dx, dy in NEIGHBORS8:
        q = (x + dx, y + dy)
        if q not in cluster and in_domain(q[0], q[1]):
            frontier.add(q)

# -----------------------------
# Tip / branching utilities (2D)
# -----------------------------
def count_occupied_neighbors(site, cluster):
    x, y = site
    c = 0
    for dx, dy in NEIGHBORS8:
        if (x + dx, y + dy) in cluster:
            c += 1
    return c

def compute_tips(cluster):
    tips = []
    for s in cluster:
        if count_occupied_neighbors(s, cluster) == 1:
            tips.append(s)
    return tips

def pick_branch_direction(theta_deg, rng):
    """
    2D analogue: choose a direction that makes angle theta from +y (vertical up).
    We pick left or right randomly.
    """
    theta = math.radians(theta_deg)
    sign = -1 if rng.random() < 0.5 else 1
    dx = sign * math.sin(theta)
    dy = math.cos(theta)
    return (dx, dy)

def preferred_site_from_tip(tip, dir_unit):
    x, y = tip
    best = None
    best_dot = -1e9
    ux, uy = dir_unit
    for dx, dy in NEIGHBORS8:
        dot = dx * ux + dy * uy
        if dot > best_dot:
            best_dot = dot
            best = (x + dx, y + dy)
    return best

def run_walker(cluster, frontier, origin, launch_radius, kill_radius, max_steps, rng, smin, smax, preferred_targets):
    """
    One walker, baseline rules:
    - start on the launch circle
    - random walk on the grid
    - stick immediately upon reaching frontier (sticking prob = 1)

    Details regarding the random walk:
    - if the walker goes beyond kill_radius, we re-inject it on the launch circle, which makes us not waste the walkers, 
    instead of just killig it and generating a new one
    """
    ox, oy = origin
    kill_r2 = kill_radius * kill_radius

    x, y = launch_point(launch_radius, origin, rng)

    for _ in range(max_steps):
        # If it wanders too far, teleport it back to the launch circle
        if sqdist(x, y, ox, oy) > kill_r2:
            x, y = launch_point(launch_radius, origin, rng)
            continue

        # Baseline sticking rule: stick when you are on a frontier site
        if (x, y) in frontier:
            xi = xi_verticalness((x, y), origin)
            if not (smin <= xi <= smax):
                dx, dy = random_step(rng)
                x += dx
                y += dy
                continue

            return (x, y)
        # Otherwise keep walking
        dx, dy = random_step(rng)
        xn, yn = x + dx, y + dy
        if not in_domain(xn, yn):
            continue
        x, y = xn, yn


    # If we hit max_steps, we give up on this walker and spawn a new one
    return None
