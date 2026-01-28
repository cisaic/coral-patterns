from typing import Set, Tuple, List
import numpy as np
from tqdm import tqdm
import random
import os
from multiprocessing import Pool

from .config import DEFAULTS
from .dla import run_walker
from .helpers import radius_from_r2

def compute_growth_probability_parallel(cluster: Set[Tuple[int, int]], max_r2: float, num_walkers: int):
    """Parallelized version"""
    origin = (0, 0)
    R = radius_from_r2(max_r2)
    launch_r = R + DEFAULTS["launch_margin"]
    kill_r = launch_r + DEFAULTS["kill_margin"]
    rng = random.Random(DEFAULTS["rng_seed"])
    
    # Prepare arguments for each walker (with unique seeds)
    walker_args = (
        cluster, origin, launch_r, kill_r, DEFAULTS["max_steps_per_walker"], 
         rng)
    
    # Use all available cores
    num_cores = os.cpu_count() - 1
    print(f"Using {num_cores} cores")
    
    growth_counts = {site: 0 for site in cluster}
    paths = []
    
    # Run walkers in parallel
    with Pool(num_cores) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_walker, walker_args),
            total=num_walkers,
            desc="Running walkers"
        ))
    
    # Aggregate results
    for growth_site, path in results:
        if growth_site is not None:
            growth_counts[growth_site] += 1
            paths.append(path)
    
    growth_probabilities = {site: count / len(paths) for site, count in growth_counts.items()}
    print(f"5 values with highest growth probabilities: {sorted(growth_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"Number of successful walkers: {len(paths)}")
    
    return growth_probabilities, paths


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
    q_steps: int
) -> Tuple[List[float], List[float]]:
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

    growth_probabilities, paths = compute_growth_probability_parallel(cluster, max_r2, num_walkers)
    print(f"first 10 growth_probabilities: {list(growth_probabilities.values())[:10]}")
    print("--------------------------------")
    print(f"last 10 growth_probabilities: {list(growth_probabilities.values())[-10:]}")
    print("--------------------------------")

    num_sites = len(cluster)
    p_vals = np.array(list(growth_probabilities.values()))

    q_vals = np.linspace(q_range[0], q_range[1], q_steps)

    sigma_q = []

    for q in q_vals:
        sum_p_q = np.sum(p_vals ** q)
        sigma = -np.log(sum_p_q) / np.log(num_sites) if sum_p_q > 0 else float("nan")
        sigma_q.append(sigma)

        print(f"q: {q}, sigma: {sigma}, sum_p_q: {sum_p_q}")

    return q_vals, sigma_q, growth_probabilities, paths