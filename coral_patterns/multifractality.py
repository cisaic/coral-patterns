from typing import Set, Tuple, List, Dict, Any
import numpy as np
from tqdm import tqdm
import random
import os
from multiprocessing import Pool
import gc
import pickle
from scipy.interpolate import interp1d

from .config import DEFAULTS
from .dla import run_walker
from .helpers import radius_from_r2, estimate_fractal_dimension

def run_single_walker(args):
    """Wrapper function for multiprocessing"""
    cluster, origin, launch_r, kill_r, max_steps, seed, record_path = args
    
    rng = random.Random(seed)
    growth_site, path = run_walker(
        cluster=cluster,
        origin=origin,
        launch_radius=launch_r,
        kill_radius=kill_r,
        max_steps=max_steps,
        rng=rng,
        record_path=record_path
    )
    return growth_site, path

def compute_growth_probability_parallel(cluster: Set[Tuple[int, int]], max_r2: float, num_walkers: int):
    """Parallelized version"""
    origin = (0, 0)
    R = radius_from_r2(max_r2)
    launch_r = R + DEFAULTS["launch_margin"]
    kill_r = launch_r + DEFAULTS["kill_margin"]
    rng = random.Random(DEFAULTS["rng_seed"])
    
    walker_args = [
        (
            cluster,
            origin,
            launch_r,
            kill_r,
            DEFAULTS["max_steps_per_walker"],
            DEFAULTS["rng_seed"] + i,
            True if i == 0 else False # record_path
        )
        for i in range(num_walkers)
    ]
    
    num_cores = os.cpu_count() - 1
    print(f"Using {num_cores} cores")
    
    growth_counts = {site: 0 for site in cluster}
    
    # Run walkers in parallel
    with Pool(num_cores) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_single_walker, walker_args, chunksize=20),
            total=num_walkers,
            desc="Running walkers"
        ))
    
    successful_walkers = 0
    sample_path = None
    # Aggregate and flush memory periodically
    for i, (growth_site, path) in enumerate(results):
        if len(path) > 0:
            sample_path = path
        if growth_site is not None:
            growth_counts[growth_site] += 1
            # paths.append(path)
            successful_walkers += 1
        
        # Every 1000 walkers, garbage collect
        if (i + 1) % 1000 == 0:
            gc.collect()
    
    growth_probabilities = {site: count / successful_walkers for site, count in growth_counts.items()}
    print(f"5 values with highest growth probabilities: {sorted(growth_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"Number of successful walkers: {successful_walkers} / {num_walkers}")
    
    # return growth_probabilities, paths
    return growth_probabilities, sample_path


def compute_growth_probability(cluster: Set[Tuple[int, int]], max_r2: float, num_walkers: int) -> float:
    """
    Calculate the growth probability of each site in the cluster.
    Given a completed cluster, launch num_walkers random walkers and measure probability of each site being selected

    """
    origin = (0, 0)
    R = radius_from_r2(max_r2)
    launch_r = R + DEFAULTS["launch_margin"]
    kill_r = launch_r + DEFAULTS["kill_margin"]

    rng = random.Random(DEFAULTS["rng_seed"])

    # initialize growth counts to 0 for all sites
    growth_counts = {site: 0 for site in cluster}
    paths = []

    for i in tqdm(range(num_walkers)):
    
        growth_site, path = run_walker(
                    cluster=cluster,
                    origin=origin,
                    launch_radius=launch_r,
                    kill_radius=kill_r,
                    max_steps=DEFAULTS["max_steps_per_walker"],
                    rng=rng,
                )
        if growth_site is not None:
            # print(f"Walker {i} grew on site {growth_site}")
            growth_counts[growth_site] += 1
            paths.append(path)

    growth_probabilities = {site: count / len(paths) for site, count in growth_counts.items()}
    # 5 values with highest growth probabilities
    print(f"5 values with highest growth probabilities: {sorted(growth_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"Number of successful walkers: {len(paths)}")

    return growth_probabilities, paths


def compute_multifractality(
    cluster: Set[Tuple[int, int]], 
    max_r2: float, 
    num_walkers: int, 
    q_range: Tuple[float, float], 
    q_steps: int,
    origin: Tuple[int, int] = (0, 0),
) -> Tuple[List[float], List[float], Dict[Tuple[int, int], float], List[Tuple[int, int]], float, float, List[float], List[float], float, float, float]:
    """
    Compute the multifractality of a completed cluster
    Where:
        growth_probability p_i is the growth probability of site i
        num_sites n is the total number of sites
        q_range is the range of q values to compute
        q_steps is the number of q values to compute
    For each q: (From DLA Scaling Laws paper Halsey, 2000)
        Sum(p_i^q) = n^(-sigma(q))
        sigma(q) = -log(Sum(p_i^q)) / log(n)
    """

    print(f"Computing multifractality for cluster with {len(cluster)} sites")
    print(f"Launching {num_walkers} random walkers...")

    # growth_probabilities, paths = compute_growth_probability_parallel(cluster, max_r2, num_walkers)
    growth_probabilities, sample_path = compute_growth_probability_parallel(cluster, max_r2, num_walkers)
    print(f"first 10 growth_probabilities: {list(growth_probabilities.values())[:10]}")
    print("--------------------------------")
    print(f"last 10 growth_probabilities: {list(growth_probabilities.values())[-10:]}")
    print("--------------------------------")

    num_sites = len(cluster)
    p_vals = np.array(list(growth_probabilities.values()))

    q_vals = np.linspace(q_range[0], q_range[1], q_steps)
    print(f"length of q_range: {len(q_range)}")

    sigma_q = []

    for q in q_vals:
        sum_p_q = np.sum(p_vals ** q)
        sigma = -np.log(sum_p_q) / np.log(num_sites) if sum_p_q > 0 else float("nan")
        sigma_q.append(sigma)

        print(f"q: {q}, sigma: {sigma}, sum_p_q: {sum_p_q}")

    multifractality_data = multifractality_fit(q_vals, sigma_q)
    slope_at_1 = multifractality_data["slope_at_1"]
    sigma_at_1 = multifractality_data["sigma_at_1"]
    q_tangent = multifractality_data["q_tangent"]
    sigma_tangent = multifractality_data["sigma_tangent"]
    slope_inf = multifractality_data["slope_inf"]
    intercept_inf = multifractality_data["intercept_inf"]
    sigma_at_3 = multifractality_data["sigma_at_3"]

    fractal_dimension = estimate_fractal_dimension(cluster, origin, max_r2=max_r2)
    print(f"Fractal dimension: {fractal_dimension}")

    # save to file
    with open(f"data/multifractality_mass-{DEFAULTS['target_mass']}_gm-{DEFAULTS['growth_mode']}_f-{DEFAULTS['friendliness']}_seed-{DEFAULTS['rng_seed']}_numwalkers-{num_walkers}.pkl", "wb") as f:
        pickle.dump((q_vals, sigma_q, growth_probabilities, sample_path, slope_at_1, sigma_at_1, q_tangent, sigma_tangent, slope_inf, intercept_inf, sigma_at_3, fractal_dimension), f)

    return {
        "q_vals": q_vals,
        "sigma_q": sigma_q,
        "growth_probabilities": growth_probabilities,
        "sample_path": sample_path,
        "slope_at_1": slope_at_1,
        "sigma_at_1": sigma_at_1,
        "q_tangent": q_tangent,
        "sigma_tangent": sigma_tangent,
        "slope_inf": slope_inf,
        "intercept_inf": intercept_inf,
        "sigma_at_3": sigma_at_3,
        "fractal_dimension": fractal_dimension,
    }

def multifractality_fit(
    q_vals: List[float],
    sigma_q: List[float],
) -> Tuple[float, float]:
    """Behaviour of the multifractality curve as:
     - as q -> infinity (slope should be linear and equal to D^2 - D)
     - Around q = 1, slope should be 1/D
     - At q = 3, sigma(q) should be equal to 1
     Where:
     - D is the fractal dimension of the cluster
    """
    
    # ==================
    # Measure slope as q -> infinity:
    # ==================
    # Fit end range of q vlaues to a linear function
    print(f"length of q_vals: {len(q_vals)}")
    print(f"length of sigma_q: {len(sigma_q)}")
    slope_inf, intercept_inf = np.polyfit(q_vals[-10:], sigma_q[-10:], deg=1)

    print(f"Slope at high q: {slope_inf:.4f}")
    print(f"Intercept at high q: {intercept_inf:.4f}")
    
    # ==================
    # Around q = 1:
    # ==================
    
    q_vals = np.array(q_vals)
    sigma_q = np.array(sigma_q)

    # Interpolate between the data points to get a smooth curve
    # Fit to cubic curve
    f = interp1d(q_vals, sigma_q, kind='cubic')

    # Compute derivative at q=1
    dq = 0.001
    slope_at_1 = (f(1 + dq) - f(1 - dq)) / (2 * dq)
    sigma_at_1 = f(1)
    
    # Tangent line: y = m(x - x0) + y0
    # y = slope_at_1 * (q - 1) + sigma_at_1
    q_tangent = np.linspace(-2, 5, 100)
    sigma_tangent = slope_at_1 * (q_tangent - 1) + sigma_at_1
    print(f"Slope at q=1: {slope_at_1:.4f}")

    # ==================
    # At q = 3:
    # ==================
    # Find the index of the value in q_range closest to 3
    idx_closest_to_3 = np.abs(q_vals - 3).argmin()
    sigma_at_3 = sigma_q[idx_closest_to_3]
    
    print(f"Sigma at q=3: {sigma_at_3:.4f}")
    
    return {
        "slope_at_1": slope_at_1,
        "sigma_at_1": sigma_at_1,
        "q_tangent": q_tangent,
        "sigma_tangent": sigma_tangent,
        "slope_inf": slope_inf,
        "intercept_inf": intercept_inf,
        "sigma_at_3": sigma_at_3
    }
