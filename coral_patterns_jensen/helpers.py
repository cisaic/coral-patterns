import math
from config import NEIGHBORS8

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
