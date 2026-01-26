# plotting.py
import math
import matplotlib.pyplot as plt

from helpers import sqdist, logspace, fit_line


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
