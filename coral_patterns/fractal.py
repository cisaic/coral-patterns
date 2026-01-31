import math


def _sqdist(x, y, ox, oy):
    dx = x - ox
    dy = y - oy
    return dx * dx + dy * dy


def mass_in_radius(cluster, origin, r):
    """Count sites within Euclidean radius r of origin."""
    r2 = float(r) * float(r)
    ox, oy = origin
    c = 0
    for (x, y) in cluster:
        if _sqdist(x, y, ox, oy) <= r2:
            c += 1
    return c


def logspace(r_lo, r_hi, n):
    """Geometric spacing between r_lo and r_hi (both > 0)."""
    if r_lo <= 0 or r_hi <= 0:
        raise ValueError("logspace requires r_lo>0 and r_hi>0.")
    if n < 2:
        return [float(r_lo)]
    la = math.log(float(r_lo))
    lb = math.log(float(r_hi))
    return [math.exp(la + (lb - la) * i / (n - 1)) for i in range(n)]


def fit_line(xs, ys):
    """Least-squares fit y = m x + c."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)

    m = num / den if den != 0 else float("nan")
    c = y_mean - m * x_mean
    return m, c


def estimate_fractal_dimension(cluster, origin, max_r2, n_points=30, fit_low=0.15, fit_high=0.65, r_min_abs=5.0):
    """
    Estimate D from M(r) ~ r^D by linear fit on log-log.
    Returns dict: D, intercept c, and sampled r_list/M_list.
    """
    R = max(10.0, math.sqrt(float(max_r2)))
    r_lo = max(float(r_min_abs), float(fit_low) * R)
    r_hi = max(r_lo * 1.05, float(fit_high) * R)

    r_list = logspace(r_lo, r_hi, int(n_points))
    M_list = [mass_in_radius(cluster, origin, r) for r in r_list]

    xs, ys = [], []
    for r, M in zip(r_list, M_list):
        if r > 0 and M > 0:
            xs.append(math.log(float(r)))
            ys.append(math.log(float(M)))

    D, c = fit_line(xs, ys)

    return {
        "D": float(D),
        "intercept": float(c),
        "r_list": r_list,
        "M_list": M_list,
        "R_used": float(R),
        "r_lo": float(r_lo),
        "r_hi": float(r_hi),
    }
