import math
from .config import NEIGHBORS8

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
    Pick a lattice point approximately on a semicircle of given radius.
    This is a way to launch walkers around the cluster.
    """
    ox, oy = origin
    theta = rng.random() * 2.0 * math.pi
    x = int(round(ox + radius * math.cos(theta)))
    y = abs(int(round(oy + radius * math.sin(theta))))
    return (x, y)


def random_step(rng):
    """One unbiased step (up/down/left/right)."""
    return rng.choice(NEIGHBORS8)

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


def normalize_probabilities(probabilities):
    """
    Normalize probabilities to sum to 1
    """
    total = probabilities.sum()
    if total == 0.0:
        return probabilities

    return (probabilities / total)


# Helper functions for neighborhood creation 
def generate_neighborhood(radius):
    """
    Generate a neighborhood of sites within a given radius
    """
    neighborhood = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            if x == 0 and y == 0:
                continue # skip the origin
            neighborhood.append((x, y))
    return neighborhood


def count_neighbors(site, cluster, neighborhood):
    """
    Count the number of neighbors of a given site in the cluster
    """
    count = 0

    # print(f"Neighborhood: {neighborhood}")

    for neighbor in neighborhood:
        candidate = (site[0] + neighbor[0], site[1] + neighbor[1])
        if candidate in cluster:
            count += 1
    return count


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