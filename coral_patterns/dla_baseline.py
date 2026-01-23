import random
import math



# Diffusion-Limited Aggregation (DLA) — baseline on a 2D lattice
#
# Explanation:
# - We start with one occupied seed at the origin.
# - We repeatedly release a random walker from a circle around the cluster.
# - The walker performs an unbiased random walk on the 4-neighborhood grid.
# - As soon as it reaches a site adjacent to the cluster, it sticks and becomes part of it.
#
# Two small tricks that were implemented to make the code more efficient :
# 1) Frontier set: we keep the set of empty sites adjacent to the cluster.
#    Then "is the walker touching the cluster?" becomes: (x,y) in frontier.
# 2) Incremental radius tracking: we store max distance^2 from the origin,
#    so we don’t rescan the whole cluster all the time



#Parameters section, you can edit them here to test different combinations

DEFAULTS = {
    "target_mass": 8000,
    "launch_margin": 6,
    "kill_margin": 25,
    "max_steps_per_walker": 80_000,
    "rng_seed": 1,
    "log_every": 1000,
}

PLOT_DEFAULTS = {
    "point_size": 0.6,
    "title_cluster": "Baseline DLA (4-neighborhood) — N=8000",
    "title_mass_radius": "Mass-radius scaling (log-log)",
    "title_mass_time": "Baseline growth: M(t)",
}


NEIGHBORS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def sqdist(x, y, ox=0, oy=0):
    """Squared distance from (x, y) to (ox, oy)."""
    dx = x - ox
    dy = y - oy
    return dx * dx + dy * dy


def sqdist_point(p, origin):
    """Squared distance from p to origin."""
    return sqdist(p[0], p[1], origin[0], origin[1])


def radius_from_r2(max_r2):
    """Convert max squared radius to an integer-ish radius."""
    return int(math.ceil(math.sqrt(max_r2)))


def launch_point(radius, origin, rng):
    """
    Pick a lattice point approximately on a circle of given radius.
    This is a way to launch walkers around the cluster.
    """
    ox, oy = origin
    theta = rng.random() * 2.0 * math.pi
    x = int(round(ox + radius * math.cos(theta)))
    y = int(round(oy + radius * math.sin(theta)))
    return (x, y)


def random_step(rng):
    """One unbiased step (up/down/left/right)."""
    return rng.choice(NEIGHBORS4)


# Frontier handling
def build_frontier(cluster):
    """
    Frontier = all empty sites adjacent (4-neighborhood) to the occupied cluster.
    If a walker lands on a frontier site, it can stick. We keep this updated throughout the simulation with the next funtion
    """
    frontier = set()
    for (x, y) in cluster:
        for dx, dy in NEIGHBORS4:
            q = (x + dx, y + dy)
            if q not in cluster:
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
    for dx, dy in NEIGHBORS4:
        q = (x + dx, y + dy)
        if q not in cluster:
            frontier.add(q)


# One random walker
def run_walker(cluster, frontier, origin, launch_radius, kill_radius, max_steps, rng):
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
            return (x, y)

        # Otherwise keep walking
        dx, dy = random_step(rng)
        x += dx
        y += dy

    # If we hit max_steps, we give up on this walker and spawn a new one
    return None

def simulate_dla(
    target_mass,
    launch_margin,
    kill_margin,
    max_steps_per_walker,
    rng_seed,
    log_every
):
    """
    Baseline isotropic DLA on a 2D lattice.

    Returns:
    - cluster (set of occupied sites)
    - origin
    - mass_history
    - max_r2 (squared radius of the farthest occupied site)
    """
    rng = random.Random(rng_seed)
    origin = (0, 0)

    cluster = {origin}
    frontier = build_frontier(cluster)

    mass_history = [1]
    max_r2 = 0  # seed is at origin -> distance^2 = 0

    while len(cluster) < target_mass:
        R = radius_from_r2(max_r2)
        launch_r = R + launch_margin
        kill_r = launch_r + kill_margin

        stuck = run_walker(
            cluster=cluster,
            frontier=frontier,
            origin=origin,
            launch_radius=launch_r,
            kill_radius=kill_r,
            max_steps=max_steps_per_walker,
            rng=rng,
        )

        if stuck is None:
            continue

        # Attach to the cluster
        cluster.add(stuck)
        update_frontier(stuck, cluster, frontier)

        # Update max radius incrementally so we don't need to do a whole full scan
        r2 = sqdist_point(stuck, origin)  # <-- fixed
        if r2 > max_r2:
            max_r2 = r2

        mass_history.append(len(cluster))

        if log_every and (len(cluster) % log_every == 0):
            print(f"[DLA] mass={len(cluster)}  R~{radius_from_r2(max_r2)}")

    return cluster, origin, mass_history, max_r2


# Measurements: M(r) and fractal dimension
def mass_in_radius(cluster, origin, r):
    """M(r): number of occupied sites within distance <= r from origin."""
    r2 = r * r
    ox, oy = origin
    count = 0
    for (x, y) in cluster:
        if sqdist(x, y, ox, oy) <= r2:
            count += 1
    return count


def logspace(a, b, n):
    """
    n points spaced evenly in log scale between a and b (a>0).
    This is useful for scaling laws.
    """
    if a <= 0 or b <= 0:
        raise ValueError("logspace requires a>0 and b>0 by definition of log so we can avoid erors here.")
    if n < 2:
        return [a]
    la = math.log(a)
    lb = math.log(b)
    return [math.exp(la + (lb - la) * i / (n - 1)) for i in range(n)]


def fit_line(xs, ys):
    """
    Fit y = m x + c by least squares.
    Returns (m, c).
    """
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    m = num / den if den != 0 else float("nan")
    c = y_mean - m * x_mean
    return m, c


def estimate_fractal_dimension(
    cluster,
    origin,
    max_r2,
    n_points=30,
    fit_low=0.15,
    fit_high=0.65,
    r_min_abs=5.0,
):
    """
    Estimate fractal dimension D from M(r) ~ r^D.

    Important :
    - Very small r: lattice / core effects.
    - Very large r near the outer radius: saturation / boundary effects.
    So we fit only on an intermediate window [fit_low*R, fit_high*R].

    Returns a dict with:
    - D (slope)
    - intercept
    - r_list, M_list (to plot later on)
    - fit indices (i0, i1)
    """
    R = max(10.0, math.sqrt(max_r2))
    r_lo = max(r_min_abs, fit_low * R)
    r_hi = max(r_lo * 1.05, fit_high * R)

    r_list = logspace(r_lo, r_hi, n_points)
    M_list = [mass_in_radius(cluster, origin, r) for r in r_list]

    # Prepare log-log data
    xs = []
    ys = []
    for r, M in zip(r_list, M_list):
        if r > 0 and M > 0:
            xs.append(math.log(r))
            ys.append(math.log(M))

    m, c = fit_line(xs, ys)

    return {
        "D": m,
        "intercept": c,
        "r_list": r_list,
        "M_list": M_list,
        "R_used": R,
        "r_lo": r_lo,
        "r_hi": r_hi,
    }

