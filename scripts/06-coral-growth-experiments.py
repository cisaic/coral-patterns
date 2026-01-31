# scripts/06-coral-growth-experiments.py

import os
import math
import random
import statistics as stats

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from coral_patterns.dla import run_walker, attach_to_cluster_neighborhood
from coral_patterns.fractal import estimate_fractal_dimension
from coral_patterns.metrics import compute_metrics
from coral_patterns.plotting_experiments import (
    plot_cluster,
    plot_D_histogram_with_errors,
    plot_mass_radius_with_fit_band,
    plot_metric_heatmap,
)

DEFAULTS = {
    "target_mass": 2000,
    "launch_margin": 6,
    "kill_margin": 25,
    "max_steps_per_walker": 80_000,
    "rng_seed": 1,
    "growth_mode": 0.0,
    "friendliness": 0.7,
    "neighborhood_radius": 1,
    "sharpness": 1.0,
    "progress": True,
    "switch_mass_fraction": None,
    "switch_mass": None,
    "growth_mode_2": None,
    "friendliness_2": None,
    "return_history": True,
}


def sqdist(x, y, ox, oy):
    dx = x - ox
    dy = y - oy
    return dx * dx + dy * dy


def sqdist_point(p, origin):
    return sqdist(p[0], p[1], origin[0], origin[1])


def radius_from_r2(max_r2):
    return int(math.ceil(math.sqrt(float(max_r2))))


def generate_neighborhood(radius):
    neigh = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            if x == 0 and y == 0:
                continue
            neigh.append((x, y))
    return neigh


def _unpack_walker_return(ret):
    """
    Compatible with run_walker implementations that return:
      - ((x, y), path)
      - (None, path)
      - (x, y)
      - None
    """
    if ret is None:
        return None

    # (something, path)
    if isinstance(ret, tuple) and len(ret) == 2:
        first = ret[0]
        if first is None:
            return None
        if isinstance(first, tuple) and len(first) == 2:
            return first

    # (x, y) directly
    if isinstance(ret, tuple) and len(ret) == 2:
        a, b = ret
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            return ret

    return None


def simulate_dla(**cfg):
    target_mass = int(cfg["target_mass"])
    launch_margin = int(cfg["launch_margin"])
    kill_margin = int(cfg["kill_margin"])
    max_steps = int(cfg["max_steps_per_walker"])
    rng_seed = int(cfg["rng_seed"])

    growth_mode = float(cfg["growth_mode"])
    friendliness = float(cfg["friendliness"])
    neighborhood_radius = int(cfg["neighborhood_radius"])
    sharpness = float(cfg["sharpness"])

    progress = bool(cfg.get("progress", True))
    return_history = bool(cfg.get("return_history", True))

    switch_mass_fraction = cfg.get("switch_mass_fraction", None)
    switch_mass = cfg.get("switch_mass", None)
    growth_mode_2 = cfg.get("growth_mode_2", None)
    friendliness_2 = cfg.get("friendliness_2", None)

    rng = random.Random(rng_seed)
    origin = (0, 0)
    neighborhood = generate_neighborhood(neighborhood_radius)

    if switch_mass is None and switch_mass_fraction is not None:
        switch_mass = int(round(float(switch_mass_fraction) * float(target_mass)))
    if switch_mass is not None:
        switch_mass = max(2, min(int(switch_mass), int(target_mass) - 1))

    def current_params(mass_now):
        gm = growth_mode
        fr = friendliness
        if switch_mass is not None and mass_now >= switch_mass:
            if growth_mode_2 is not None:
                gm = float(growth_mode_2)
            if friendliness_2 is not None:
                fr = float(friendliness_2)
        return gm, fr

    cluster = {origin}
    cluster_history = [origin] if return_history else []
    max_r2 = 0.0

    with tqdm(total=target_mass, disable=not progress) as pbar:
        pbar.update(1)
        while len(cluster) < target_mass:
            R = radius_from_r2(max_r2)
            launch_r = R + launch_margin
            kill_r = launch_r + kill_margin

            ret = run_walker(
                cluster=cluster,
                origin=origin,
                launch_radius=launch_r,
                kill_radius=kill_r,
                max_steps=max_steps,
                rng=rng,
                record_path=False,
            )
            stuck = _unpack_walker_return(ret)
            if stuck is None:
                continue

            gm, fr = current_params(len(cluster))

            new_site = attach_to_cluster_neighborhood(
                cluster=cluster,
                growth_site=stuck,
                growth_mode=gm,
                friendliness=fr,
                neighborhood=neighborhood,
                sharpness=sharpness,
                rng=rng,
            )
            if new_site is None:
                continue

            cluster.add(new_site)
            if return_history:
                cluster_history.append(new_site)

            r2 = float(sqdist_point(new_site, origin))
            if r2 > max_r2:
                max_r2 = r2

            pbar.update(1)

    return cluster, cluster_history, origin, max_r2


def summarize(values):
    n = len(values)
    mean = stats.mean(values)
    std = stats.stdev(values) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else float("nan")
    ci95 = 1.96 * sem if n > 1 else float("nan")
    return {"n": n, "mean": float(mean), "std": float(std), "sem": float(sem), "ci95": float(ci95)}


def run_one_seed(seed, cfg):
    cfg_local = dict(cfg)
    cfg_local["rng_seed"] = int(seed)
    cfg_local["progress"] = False
    cfg_local["return_history"] = False

    cluster, _, origin, max_r2 = simulate_dla(**cfg_local)
    out = estimate_fractal_dimension(cluster, origin, max_r2=max_r2)
    met = compute_metrics(cluster, origin, max_r2)

    return {
        "seed": int(seed),
        "D": float(out["D"]),
        "c": float(out["intercept"]),
        "metrics": met,
    }


def multi_seed_run(cfg, seeds, parallel=True, max_workers=None):
    results = []
    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_one_seed, s, cfg) for s in seeds]
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for s in seeds:
            results.append(run_one_seed(s, cfg))

    results.sort(key=lambda d: d["seed"])
    return results


def reference_configs(base_cfg):
    return {
        "massive": {
            **base_cfg,
            "growth_mode": 0.0,
            "friendliness": 0.85,
            "sharpness": 1.0,
        },
        "cauliflower": {
            **base_cfg,
            "growth_mode": -1.0,
            "friendliness": 0.85,
            "sharpness": 1.2,
        },
        "columnar": {
            **base_cfg,
            "growth_mode": 1.0,
            "friendliness": 1.0,
            "sharpness": 1.0,
        },
        "table_transition": {
            **base_cfg,
            "growth_mode": 0.75,
            "friendliness": 0.70,
            "switch_mass_fraction": 0.55,
            "growth_mode_2": -0.95,
            "friendliness_2": 0.90,
            "sharpness": 1.1,
        },
    }


def sensitivity_grid(base_cfg, growth_modes, friendliness_vals, seeds_per_cell=4, parallel=True):
    D_mean = np.zeros((len(friendliness_vals), len(growth_modes)))
    vert_mean = np.zeros_like(D_mean)
    tips_mean = np.zeros_like(D_mean)

    seeds = list(range(1, seeds_per_cell + 1))

    for j, fr in enumerate(friendliness_vals):
        for i, gm in enumerate(growth_modes):
            cfg = dict(base_cfg)
            cfg["growth_mode"] = float(gm)
            cfg["friendliness"] = float(fr)
            cfg["switch_mass_fraction"] = None
            cfg["growth_mode_2"] = None
            cfg["friendliness_2"] = None

            res = multi_seed_run(cfg, seeds, parallel=parallel)
            Ds = [r["D"] for r in res]
            verts = [r["metrics"]["verticality"] for r in res]
            tipsf = [r["metrics"]["tips_frac"] for r in res]

            D_mean[j, i] = float(np.mean(Ds))
            vert_mean[j, i] = float(np.mean(verts))
            tips_mean[j, i] = float(np.mean(tipsf))

    return D_mean, vert_mean, tips_mean


def main():
    base = "plots/coral_growth"
    out_struct = os.path.join(base, "structures")
    out_power = os.path.join(base, "powerlaw")
    out_dist = os.path.join(base, "distributions")
    out_heat = os.path.join(base, "heatmaps")
    os.makedirs(out_struct, exist_ok=True)
    os.makedirs(out_power, exist_ok=True)
    os.makedirs(out_dist, exist_ok=True)
    os.makedirs(out_heat, exist_ok=True)

    K_SEEDS = 20
    SEEDS = list(range(1, K_SEEDS + 1))
    PARALLEL = True
    MAX_WORKERS = None

    BASE_CFG = dict(DEFAULTS)
    BASE_CFG["target_mass"] = 2500
    BASE_CFG["progress"] = True

    rep_seed = 11  # fixed seed for reproducibility

    print("\n=== Running 4 reference configs (multi-seed) ===")
    configs = reference_configs(BASE_CFG)

    for name, cfg in configs.items():
        print(f"\n--- {name} ---")
        res = multi_seed_run(cfg, SEEDS, parallel=PARALLEL, max_workers=MAX_WORKERS)

        Ds = [r["D"] for r in res]
        cs = [r["c"] for r in res]
        summD = summarize(Ds)

        print(
            f"K={summD['n']}  D_mean={summD['mean']:.4f}  std={summD['std']:.4f}  "
            f"SEM={summD['sem']:.4f}  95%CI=±{summD['ci95']:.4f}"
        )

        # Distributions
        plot_D_histogram_with_errors(
            Ds,
            mean=summD["mean"],
            ci95=summD["ci95"],
            title=f"Distribution of D: {name}",
            savepath=os.path.join(out_dist, f"distribution_{name}.png"),
            bins="fd",
        )

        # Rerun representative seed with history
        cfg_rep = dict(cfg)
        cfg_rep["rng_seed"] = rep_seed
        cfg_rep["progress"] = False
        cfg_rep["return_history"] = True

        cluster, history, origin, max_r2 = simulate_dla(**cfg_rep)
        out = estimate_fractal_dimension(cluster, origin, max_r2=max_r2)
        Dc_pairs = list(zip(Ds, cs))

        # Structures
        plot_cluster(
            history,
            title=f"Structure: {name} (seed={rep_seed})",
            savepath=os.path.join(out_struct, f"structure_{name}_seed{rep_seed}.png"),
            point_size=0.6,
        )

        # Power law (log-log)
        plot_mass_radius_with_fit_band(
            out,
            Dc_pairs,
            title=f"Power law (log-log): {name} (seed={rep_seed})",
            savepath=os.path.join(out_power, f"powerlaw_{name}_seed{rep_seed}.png"),
        )

        vert = float(np.mean([r["metrics"]["verticality"] for r in res]))
        tipsf = float(np.mean([r["metrics"]["tips_frac"] for r in res]))
        comp = float(np.mean([r["metrics"]["compactness"] for r in res]))
        print(f"avg verticality={vert:.3f}  avg tips_frac={tipsf:.4f}  avg compactness={comp:.4f}")

    # Heatmaps
    print("\n=== Sensitivity grid (growth_mode x friendliness) ===")
    GRID_CFG = dict(BASE_CFG)
    GRID_CFG["target_mass"] = 1600
    GRID_CFG["progress"] = False
    GRID_CFG["return_history"] = False

    growth_modes = np.linspace(-1.0, 1.0, 7)
    friendliness_vals = np.linspace(0.0, 1.0, 7)

    D_mean, vert_mean, tips_mean = sensitivity_grid(
        GRID_CFG,
        growth_modes=growth_modes,
        friendliness_vals=friendliness_vals,
        seeds_per_cell=4,
        parallel=PARALLEL,
    )

    plot_metric_heatmap(
        growth_modes, friendliness_vals, D_mean,
        title="Heatmap: mean fractal dimension D",
        xlabel="growth_mode (horizontal → vertical)",
        ylabel="friendliness (branchy → compact)",
        savepath=os.path.join(out_heat, "heatmap_D_mean.png"),
    )
    plot_metric_heatmap(
        growth_modes, friendliness_vals, vert_mean,
        title="Heatmap: mean verticality (height/width)",
        xlabel="growth_mode (horizontal → vertical)",
        ylabel="friendliness (branchy → compact)",
        savepath=os.path.join(out_heat, "heatmap_verticality_mean.png"),
    )
    plot_metric_heatmap(
        growth_modes, friendliness_vals, tips_mean,
        title="Heatmap: mean tips fraction",
        xlabel="growth_mode (horizontal → vertical)",
        ylabel="friendliness (branchy → compact)",
        savepath=os.path.join(out_heat, "heatmap_tipsfrac_mean.png"),
    )

    print(f"\nDone. Outputs saved under: {base}/")


if __name__ == "__main__":
    main()

