import random
from tqdm import tqdm
from typing import Set, Tuple, List, Any
import pickle

from .helpers import (
    radius_from_r2,
    sqdist_point,
    generate_neighborhood, 
)
from .dla import (
    run_walker,
    attach_to_cluster_neighborhood
)

def simulate_dla(
    target_mass: int,
    launch_margin: float,
    kill_margin: float,
    max_steps_per_walker: int,
    rng_seed: int,
    log_every: int,
    growth_mode: float,
    friendliness: float,
    neighborhood_radius: int,
    sharpness: float,
) -> Tuple[Set[Tuple[int, int]], List[Tuple[int, int]], Tuple[int, int], List[int], int]:
    """
    Isotropic DLA on a 2D lattice.

    Returns:
    - cluster (set of occupied sites)
    - origin
    - mass_history
    - max_r2 (squared radius of the farthest occupied site)
    """
    rng = random.Random(rng_seed)
    origin = (0, 0) # TODO: Make origin configurable

    neighborhood = generate_neighborhood(neighborhood_radius)

    cluster = {origin}
    cluster_history = [origin]
    attachment_counts = {origin: 0}

    mass_history = [1]
    max_r2 = 0  # seed is at origin -> distance^2 = 0

    with tqdm(total=target_mass) as pbar:

        while len(cluster) < target_mass:
            R = radius_from_r2(max_r2)
            launch_r = R + launch_margin
            kill_r = launch_r + kill_margin

            growth_site, _ = run_walker(
                cluster=cluster,
                origin=origin,
                launch_radius=launch_r,
                kill_radius=kill_r,
                max_steps=max_steps_per_walker,
                rng=rng,
            )
            # print(f"Growing from: {growth_site}")

            if growth_site is None:
                continue

            # attach to cluster neighborhood with probability
            new_site = attach_to_cluster_neighborhood(
                cluster, growth_site, growth_mode, friendliness, neighborhood, sharpness, rng
            )

            if new_site is None:
                continue

            # Attach to the cluster
            cluster.add(new_site)
            cluster_history.append(new_site)
            pbar.update(1)
            
            # Update max radius incrementally so we don't need to do a whole full scan
            r2 = sqdist_point(new_site, origin)  
            if r2 > max_r2:
                max_r2 = r2

            mass_history.append(len(cluster))

            if log_every and (len(cluster) % log_every == 0):
                print(f"[DLA] mass={len(cluster)}  R~{radius_from_r2(max_r2)}")

    # save cluster, cluster_history, origin, mass_history, max_r2 to file
    with open(f"data/dla_mass-{target_mass}_gm-{growth_mode}_f-{friendliness}_seed-{rng_seed}.pkl", "wb") as f:
        pickle.dump((cluster, cluster_history, origin, mass_history, max_r2), f)

    return cluster, cluster_history, origin, mass_history, max_r2
