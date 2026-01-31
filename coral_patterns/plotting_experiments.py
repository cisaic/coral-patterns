import os
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_cluster(cluster_history, title, savepath, point_size=0.6):
    """Cluster image (structure)."""
    _ensure_dir(savepath)

    xs = [x for (x, _) in cluster_history]
    ys = [y for (_, y) in cluster_history]
    history = np.linspace(0, 1, len(cluster_history))

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=point_size, c=history, cmap="cool", vmin=0, vmax=1)
    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    plt.title(title)
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_D_histogram_with_errors(Ds, mean, ci95, title, savepath, bins="fd"):
    """Distribution of D with Poisson errorbars on histogram counts + CI band."""
    _ensure_dir(savepath)

    Ds = np.array(Ds, dtype=float)
    counts, edges = np.histogram(Ds, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    yerr = np.sqrt(counts)

    plt.figure(figsize=(7, 4))
    plt.hist(Ds, bins=edges, alpha=0.7)
    plt.errorbar(centers, counts, yerr=yerr, fmt="none", capsize=3)

    plt.axvline(mean, linestyle="--", label=f"mean={mean:.3f}")
    if np.isfinite(ci95):
        plt.axvspan(mean - ci95, mean + ci95, alpha=0.2, label=f"95% CI ±{ci95:.3f}")

    plt.xlabel("Estimated fractal dimension D")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_mass_radius_with_fit_band(sample_out, Dc_pairs, title, savepath):
    """
    Power-law plot (log-log) + uncertainty band from covariance of (D,c).
    log M = c + D log r
    """
    _ensure_dir(savepath)

    r_list = np.array(sample_out["r_list"], dtype=float)
    M_list = np.array(sample_out["M_list"], dtype=float)

    Ds = np.array([p[0] for p in Dc_pairs], dtype=float)
    cs = np.array([p[1] for p in Dc_pairs], dtype=float)

    mu_D = float(Ds.mean())
    mu_c = float(cs.mean())

    cov = np.cov(np.vstack([Ds, cs]), ddof=1) if len(Ds) > 1 else np.zeros((2, 2))
    log_r = np.log(r_list)

    mean_logM = mu_c + mu_D * log_r
    var_logM = cov[1, 1] + (log_r ** 2) * cov[0, 0] + 2.0 * log_r * cov[0, 1]
    std_logM = np.sqrt(np.maximum(var_logM, 0.0))

    z = 1.96
    lo = np.exp(mean_logM - z * std_logM)
    hi = np.exp(mean_logM + z * std_logM)
    mean_fit = np.exp(mean_logM)

    plt.figure(figsize=(7, 5))
    plt.loglog(r_list, M_list, marker="o", linestyle="None", label="M(r) (one seed)")
    plt.loglog(r_list, mean_fit, linestyle="--", label=f"ensemble mean fit (D≈{mu_D:.3f})")
    if len(Ds) > 1:
        plt.fill_between(r_list, lo, hi, alpha=0.2, label="~95% band (from cov(D,c))")

    plt.xlabel("r")
    plt.ylabel("M(r)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_heatmap(grid_x, grid_y, Z, title, xlabel, ylabel, savepath):
    """Heatmap for a metric over (growth_mode, friendliness)."""
    _ensure_dir(savepath)

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[min(grid_x), max(grid_x), min(grid_y), max(grid_y)],
        interpolation="nearest",
    )
    plt.colorbar(im)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(False)
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()
