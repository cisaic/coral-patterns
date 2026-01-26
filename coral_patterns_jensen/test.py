import random
import math
import matplotlib.pyplot as plt

# Grid/runtime parameters
DEFAULTS = {
    "target_mass": 4000,
    "launch_margin": 6,
    "kill_margin": 25,
    "max_steps_per_walker": 80_000,
    "rng_seed": 1,
    "log_every": 2000,
}

# Neighborhood definition
NEIGHBORS8 = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]

PAPER_PARAMS = {
    "smin": 0.0,
    "smax": 0.99,              # min/max values of s
    "nu_mm_per_year": 5.0,     # elongation rate (mm/year)
    "delta_sub_mm": 5.0,       # sub division distance
    "l_br_mm": 10.0,           # interbranching length
    "theta_deg": 55.0,         # branching angle         
}
# helper funvtions 

def in_domain(x, y):
    """Ensure that the coral grows upward"""
    return (y >= 0)

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
    if y < 0: y = -y

    return (x, y)

def random_step(rng):
    """One unbiased step (up/down/left/right)."""
    return rng.choice(NEIGHBORS8)


# Frontier handling
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

# -----------------------------
# Direction-selective sticking score (2D proxy)
# -----------------------------
def xi_verticalness(site, origin):
    """
    Proxy for paper's direction score.
    1 means strongly vertical, 0 means horizontal.
    """
    x, y = site
    ox, oy = origin
    rx = x - ox
    ry = y - oy
    norm = math.sqrt(rx * rx + ry * ry) + 1e-12
    return abs(ry / norm)

# One random walker
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


def simulate_dla_with_paper_params(cfg, paper):
    rng = random.Random(cfg["rng_seed"])
    origin = (0, 0)

    cluster = {origin}
    frontier = build_frontier(cluster)

    max_r2 = 0
    mass_history = [1]

    # physical time tracking
    nu = float(paper["nu_mm_per_year"])
    delta = float(paper["delta_sub_mm"])
    dt_years = (delta / nu) if nu > 0 else 0.0
    time_history_years = [0.0]
    t_years = 0.0

    # branching setup
    l_br_mm = float(paper["l_br_mm"])
    l_br_steps = max(1, int(round(l_br_mm / max(delta, 1e-9))))
    theta_deg = float(paper["theta_deg"])

    # growth mode window
    smin = float(paper["smin"])
    smax = float(paper["smax"])

    leaders = []
    preferred_targets = set()

    def refresh_preferred_targets():
        preferred_targets.clear()
        for L in leaders:
            tip = L["tip"]
            target = preferred_site_from_tip(tip, L["dir"])
            if target in frontier:
                preferred_targets.add(target)

    while len(cluster) < cfg["target_mass"]:
        R = radius_from_r2(max_r2)
        launch_r = R + cfg["launch_margin"]
        kill_r = launch_r + cfg["kill_margin"]

        if (len(cluster) > 1) and (len(cluster) % l_br_steps == 0):
            tips = compute_tips(cluster)
            if tips:
                tip = tips[rng.randrange(len(tips))]
                leaders.append({"tip": tip, "dir": pick_branch_direction(theta_deg, rng)})
                refresh_preferred_targets()

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

        cluster.add(stuck)
        update_frontier(stuck, cluster, frontier)

        for L in leaders:
            if count_occupied_neighbors(L["tip"], cluster) != 1 and count_occupied_neighbors(stuck, cluster) == 1:
                L["tip"] = stuck
        refresh_preferred_targets()

        r2 = sqdist_point(stuck, origin)
        if r2 > max_r2:
            max_r2 = r2

        mass_history.append(len(cluster))

        t_years += dt_years
        time_history_years.append(t_years)

        if cfg["log_every"] and (len(cluster) % cfg["log_every"] == 0):
            print(f"[DLA+paper] mass={len(cluster)}  R~{radius_from_r2(max_r2)}  t~{t_years:.2f} yr")

    return cluster, origin, mass_history, time_history_years, max_r2

# -----------------------------
# Plotting
# -----------------------------
def plot_cluster(cluster, title="DLA cluster", point_size=0.6):
    xs = [x for (x, _) in cluster]
    ys = [y for (_, y) in cluster]
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=point_size)
    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    plt.title(title)
    plt.show()

def plot_mass_vs_time(mass_history, time_years, title="Mass vs time"):
    plt.figure(figsize=(7, 4))
    plt.plot(time_years, mass_history)
    plt.xlabel("time (years)")
    plt.ylabel("mass M")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

# Measurements: M(r) and fractal dimension
def mass_in_radius(cluster, origin, r):
    """M(r): number of occupied sites within distance <= r from origin."""
    r2 = r * r
    ox, oy = origin
    c = 0
    for (x, y) in cluster:
        if sqdist(x, y, ox, oy) <= r2:
            c += 1
    return c

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
    xm = sum(xs) / n
    ym = sum(ys) / n
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    m = num / den if den != 0 else float("nan")
    c = ym - m * xm
    return m, c

def plot_mass_radius(cluster, origin, max_r2, title="Mass-radius scaling"):
    R = max(10.0, math.sqrt(max_r2))
    r_lo = max(5.0, 0.15 * R)
    r_hi = max(r_lo * 1.05, 0.65 * R)

    r_list = logspace(r_lo, r_hi, 30)
    M_list = [mass_in_radius(cluster, origin, r) for r in r_list]

    xs, ys = [], []
    for r, M in zip(r_list, M_list):
        if r > 0 and M > 0:
            xs.append(math.log(r))
            ys.append(math.log(M))

    D, c = fit_line(xs, ys)
    M_fit = [math.exp(c) * (r ** D) for r in r_list]

    plt.figure(figsize=(6, 5))
    plt.loglog(r_list, M_list, marker="o", linestyle="None", label="M(r)")
    plt.loglog(r_list, M_fit, linestyle="-", label=f"fit slope D ≈ {D:.3f}")
    plt.xlabel("r")
    plt.ylabel("M(r)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.show()

# -----------------------------
# Example run (WITH plotting)
# -----------------------------
if __name__ == "__main__":
    cfg = DEFAULTS.copy()

    cluster, origin, mass_hist, time_hist_years, max_r2 = simulate_dla_with_paper_params(cfg, PAPER_PARAMS)

    print("Done. Final mass:", len(cluster), "Final time (years):", time_hist_years[-1])

    plot_cluster(cluster, title="DLA + paper-like controls (2D)", point_size=0.6)
    plot_mass_vs_time(mass_hist, time_hist_years, title="Mass vs physical time (years)")
    plot_mass_radius(cluster, origin, max_r2, title="Mass-radius scaling (log-log)")
