import random
import math
from tqdm import tqdm
import numpy as np

# Diffusion-Limited Aggregation (DLA) — baseline on a 2D lattice
#
# Explanation:
# - We start with one occupied seed at the origin ("on the ground")
# - We repeatedly release a random walker from a semi-circle around the cluster (walkers not launched from below)
# - The walker performs an unbiased random walk on the 8-neighborhood grid.
# - As soon as it reaches a site adjacent to the cluster, it sticks with a given probability
# - Probabilities are parameterized by:
# -- growth_mode (preference for horizontal vs vertical growth) 
# -- friendliness (preference for sites with more or fewer neighbors) 
#
# One small trick implemented to make the code more efficient :
# - Incremental radius tracking: we store max distance^2 from the origin, so we don’t rescan the whole cluster all the time



#Parameters section, you can edit them here to test different combinations

DEFAULTS = {
    "target_mass": 3000,
    "launch_margin": 6,
    "kill_margin": 25,
    "max_steps_per_walker": 80_000,
    "rng_seed": 1,
    "log_every": 200,
    "growth_mode": 0.99,
    "friendliness": 1,
    "neighborhood_radius": 2,
    "sharpness": 2,
}

PLOT_DEFAULTS = {
    "point_size": 0.6,
    "title_cluster": "Parameterized DLA (8-neighborhood) — N=1000",
    "title_mass_radius": "Mass-radius scaling (log-log)",
    "title_mass_time": "Baseline growth: M(t)",
}


NEIGHBORS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
NEIGHBORS8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
VALID_NEIGHBORS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0)] 


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


# One random walker
def run_walker(cluster, origin, launch_radius, kill_radius, max_steps, rng):
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

        # Baseline sticking rule: stick when you are on a cluster site
        if (x, y) in cluster:
            return (x, y)

        # Otherwise keep walking
        dx, dy = random_step(rng)
        x += dx
        y += dy

    # If we hit max_steps, we give up on this walker and spawn a new one
    return None

def find_available_neighbors(cluster, stuck):
    """
    Return the index of available neighbors for a given stuck site that are not in the cluster
    """
    available_neighbors_indices = []
    available_neighbors = []
    candidate_neighbors = VALID_NEIGHBORS.copy()
    for neighbor in candidate_neighbors:
        candidate = (stuck[0] + neighbor[0], stuck[1] + neighbor[1])
        if candidate not in cluster:
            available_neighbors_indices.append(candidate_neighbors.index(neighbor))
            available_neighbors.append(candidate)
    
    # print(f"Available neighbors: {available_neighbors}")
    # print(f"Available neighbors indices: {available_neighbors_indices}")
        
    return available_neighbors_indices, available_neighbors


def growth_mode_probability(cluster, stuck, growth_mode, available_neighbors_indices):
    """
    Probability distribution of neighbors to attach to given parameter: growth_mode (range -1 to 1)
    growth_mode = -1: Prefer horizontal growth
    growth_mode = 1: Prefer vertical growth
    growth_mode = 0: Uniform

    growth_mode in [-1, 1]
        -1.0 = horizontal only (left/right)
        -0.5 = sides + top-diagonals equal (no top)
        0.0  = all 5 allowed directions equal (no bottom)
        0.5 = top + top-diagonals equal (no sides)
        1.0  = top only
    """

    assert growth_mode >= -1.0 and growth_mode <= 1.0

    weights_horizontal     = np.array([1.0, 0.0, 0.0, 0.0, 1.0])  # growth_mode= -1.0 -> horizontal only (left/right)
    weights_diagonal_sides = np.array([1.0, 1.0, 0.0, 1.0, 1.0])  # growth_mode= -0.5 -> sides + top-diagonals equal (no top)
    weights_uniform        = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # growth_mode= 0.0 -> all 5 allowed directions are equal (no bottom)
    weights_diagonal_top   = np.array([0.0, 1.0, 1.0, 1.0, 0.0])  # growth_mode= 0.5 -> top + top-diagonals equal (no sides)
    weights_vertical       = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # growth_mode= 1.0 -> top only

    if growth_mode <= -0.5:
        # Interpolate between weights_horizontal (at -1) and weights_diagonal_sides (at -0.5)
        interpolation = (growth_mode + 1.0) / 0.5  # t=0 at -1, t=1 at -0.5
        probabilities = (1 - interpolation) * weights_horizontal + interpolation * weights_diagonal_sides
    elif growth_mode <= 0.0:
        # Interpolate between weights_diagonal_sides (at -0.5) and weights_uniform (at 0)
        interpolation = (growth_mode + 0.5) / 0.5  # t=0 at -0.5, t=1 at 0
        probabilities = (1 - interpolation) * weights_diagonal_sides + interpolation * weights_uniform
    elif growth_mode < 0.5:
        # Interpolate between weights_uniform (at 0) and weights_diagonal_top (at 0.5)
        interpolation = (growth_mode - 0.0) / 0.5  # t=0 at 0, t=1 at 0.5
        probabilities = (1 - interpolation) * weights_uniform + interpolation * weights_diagonal_top
    else:
        # Interpolate between weights_diagonal_top (at 0.5) and weights_vertical (at 1)
        interpolation = (growth_mode - 0.5) / 0.5  # t=0 at 0.5, t=1 at 1
        probabilities = (1 - interpolation) * weights_diagonal_top + interpolation * weights_vertical
    
    probabilities = normalize_probabilities(probabilities)
    return probabilities

def mask_probabilities(probabilities, available_neighbors_indices):
    """
    Mask out unavailable neighbors
    """
    probabilities_masked = probabilities.copy()
    all_indices = set(range(5))
    unavailable = all_indices - set(available_neighbors_indices)
    probabilities_masked[list(unavailable)] = 0.0
    return probabilities_masked

def normalize_probabilities(probabilities):
    """
    Normalize probabilities to sum to 1
    """
    total = probabilities.sum()
    if total == 0.0:
        return probabilities

    return (probabilities / total)

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

def friendliness_probability(cluster, stuck, friendliness, available_neighbors, available_neighbors_indices, neighborhood, sharpness):
    """
    Calculate the probabilities of attaching to a given site based on the number of neighbors
    Friendliness controls preference for attaching to sites with more or less neighbors
    friendliness = 0: prefer sites with fewer neighbors
    friendliness = 0.5: no preference
    friendliness = 1: prefer sites with more neighbors
    """

    assert friendliness >= 0.0 and friendliness <= 1.0

    neighbor_counts = np.zeros(len(VALID_NEIGHBORS))
    for neighbor, idx in zip(available_neighbors, available_neighbors_indices):
        neighbor_counts[idx] = count_neighbors(neighbor, cluster, neighborhood)

    if neighbor_counts.sum() == 0:
        return np.zeros(len(VALID_NEIGHBORS))
    
    # Normalize to [0, 1]
    min_count = neighbor_counts[neighbor_counts != 0].min()
    max_count = neighbor_counts.max()
    if max_count == min_count:
        # avoid dividing by 0
        neighbor_scores = np.zeros_like(neighbor_counts) / len(neighbor_counts)
    else:
        neighbor_scores = (neighbor_counts - min_count) / (max_count - min_count)

    neighbor_inverse = 1.0 - neighbor_scores
    probabilities = (1 - friendliness) * neighbor_inverse + friendliness * neighbor_scores

    probabilities[neighbor_counts == 0] = 0.0
    probabilities = np.power(probabilities, sharpness)

    probabilities = normalize_probabilities(probabilities)

    # print(f"Neighbor counts: {neighbor_counts}")
    # print(f"Neighbor scores: {neighbor_scores}")
    # print(f"Neighbor inverse: {neighbor_inverse}")
    # print(f"Friendliness probabilities: {probabilities}")

    # print("--------------------------------")
    return probabilities

def attach_to_cluster_neighborhood(cluster, stuck, growth_mode, friendliness, neighborhood, sharpness, rng):
    """
    Attach to the cluster neighborhood with parameterized probabilities
    """
    available_neighbors_indices, available_neighbors = find_available_neighbors(cluster, stuck)
    if len(available_neighbors) == 0:
        return None

    growth_mode_probabilities = growth_mode_probability(cluster, stuck, growth_mode, available_neighbors_indices)
    neighbor_probabilities = friendliness_probability(cluster, stuck, friendliness, available_neighbors, available_neighbors_indices, neighborhood, sharpness)

    combined_probabilities = growth_mode_probabilities + neighbor_probabilities

    attachment_probs = mask_probabilities(combined_probabilities, available_neighbors_indices)
    attachment_probs = normalize_probabilities(attachment_probs).tolist()

    # edge case where all probabilities are 0
    if sum(attachment_probs) == 0:
        return None

    # select a new site based on the probabilities
    selected_neighbor = rng.choices(VALID_NEIGHBORS, weights=attachment_probs, k=1)[0]
    new_site = (stuck[0] + selected_neighbor[0], stuck[1] + selected_neighbor[1])
    return new_site
    
def simulate_dla(
    target_mass,
    launch_margin,
    kill_margin,
    max_steps_per_walker,
    rng_seed,
    log_every,
    growth_mode,
    friendliness,
    neighborhood_radius,
    sharpness,
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

    neighborhood = generate_neighborhood(neighborhood_radius)

    cluster = {origin}
    # frontier = build_frontier(cluster)

    mass_history = [1]
    max_r2 = 0  # seed is at origin -> distance^2 = 0

    with tqdm(total=target_mass) as pbar:

        while len(cluster) < target_mass:
            R = radius_from_r2(max_r2)
            launch_r = R + launch_margin
            kill_r = launch_r + kill_margin

            stuck = run_walker(
                cluster=cluster,
                origin=origin,
                launch_radius=launch_r,
                kill_radius=kill_r,
                max_steps=max_steps_per_walker,
                rng=rng,
            )
            # print(f"Growing from: {stuck}")

            if stuck is None:
                continue

            # attach to cluster neighborhood with probability
            new_site = attach_to_cluster_neighborhood(cluster, stuck, growth_mode, friendliness, neighborhood, sharpness, rng)

            if new_site is None:
                continue

            # Attach to the cluster
            cluster.add(new_site)
            pbar.update(1)
            # print("")
            
            # Update max radius incrementally so we don't need to do a whole full scan
            r2 = sqdist_point(new_site, origin)  
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

