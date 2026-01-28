import math
import numpy as np
from typing import Any, List, Tuple, Set, Optional, Sequence
import random

from .config import VALID_NEIGHBORS
from .helpers import (
    sqdist,
    random_step,
    launch_point,
    normalize_probabilities,
    count_neighbors
)

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


def run_walker(
    cluster: Set[Tuple[int, int]],
    origin: Tuple[int, int],
    launch_radius: float,
    kill_radius: float,
    max_steps: int,
    rng: random.Random, 
    record_path: bool = False
) -> Optional[Tuple[int, int]]:
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

    path = [(x, y)] if record_path else []

    for _ in range(max_steps):
        # If it wanders too far, teleport it back to the launch circle
        if sqdist(x, y, ox, oy) > kill_r2:
            x, y = launch_point(launch_radius, origin, rng)
            continue

        # Baseline sticking rule: stick when you are on a cluster site
        if (x, y) in cluster:
            return (x, y), path

        # Otherwise keep walking
        dx, dy = random_step(rng)
        x += dx
        y += dy
        if record_path:
            path.append((x, y))

    # If we hit max_steps, we give up on this walker and spawn a new one
    return None, path

def find_available_neighbors(
    cluster: Set[Tuple[int, int]],
    growth_site: Tuple[int, int]
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Return the index of available neighbors for a given growth_site site that are not in the cluster
    """
    # initialize potential neighbor only upward and sideways growth allowed
    available_neighbors_indices = []
    available_neighbors = []
    candidate_neighbors = VALID_NEIGHBORS.copy() 

    # check allowed directions within neighborhood
    for neighbor in candidate_neighbors:
        candidate = (growth_site[0] + neighbor[0], growth_site[1] + neighbor[1])

        # only the neighbors not in the cluster are selected
        if candidate not in cluster:
            available_neighbors_indices.append(candidate_neighbors.index(neighbor))
            available_neighbors.append(candidate)
    
    return available_neighbors_indices, available_neighbors

def growth_mode_probability(
    growth_mode: float
) -> np.ndarray:
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

    # Check boundaries for the growth mode.
    assert growth_mode >= -1.0 and growth_mode <= 1.0

    weights_horizontal     = np.array([1.0, 0.0, 0.0, 0.0, 1.0])  # growth_mode= -1.0 -> horizontal only (left/right)
    weights_diagonal_sides = np.array([1.0, 1.0, 0.0, 1.0, 1.0])  # growth_mode= -0.5 -> sides + top-diagonals equal (no top)
    weights_uniform        = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # growth_mode= 0.0 -> all 5 allowed directions are equal (no bottom)
    weights_diagonal_top   = np.array([0.0, 1.0, 1.0, 1.0, 0.0])  # growth_mode= 0.5 -> top + top-diagonals equal (no sides)
    weights_vertical       = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # growth_mode= 1.0 -> top only

    if growth_mode <= -0.5:
        # Interpolate between weights_horizontal (at -1) and weights_diagonal_sides (at -0.5)
        interpolation = (growth_mode + 1.0) / 0.5  # interpolation=0 at -1, interpolation=1 at -0.5
        probabilities = (1 - interpolation) * weights_horizontal + interpolation * weights_diagonal_sides
    elif growth_mode <= 0.0:
        # Interpolate between weights_diagonal_sides (at -0.5) and weights_uniform (at 0)
        interpolation = (growth_mode + 0.5) / 0.5  # interpolation=0 at -0.5, interpolation=1 at 0
        probabilities = (1 - interpolation) * weights_diagonal_sides + interpolation * weights_uniform
    elif growth_mode < 0.5:
        # Interpolate between weights_uniform (at 0) and weights_diagonal_top (at 0.5)
        interpolation = (growth_mode - 0.0) / 0.5  # interpolation=0 at 0, interpolation=1 at 0.5
        probabilities = (1 - interpolation) * weights_uniform + interpolation * weights_diagonal_top
    else:
        # Interpolate between weights_diagonal_top (at 0.5) and weights_vertical (at 1)
        interpolation = (growth_mode - 0.5) / 0.5  # interpolation=0 at 0.5, interpolation=1 at 1
        probabilities = (1 - interpolation) * weights_diagonal_top + interpolation * weights_vertical
    return normalize_probabilities(probabilities)

def mask_probabilities(
    probabilities: np.ndarray,
    available_neighbors_indices: Sequence[int]
) -> np.ndarray:
    """
    Mask out unavailable neighbors, assuming the available neighborhood
    """
    probabilities_masked = probabilities.copy()
    all_indices = set(range(len(VALID_NEIGHBORS)))

    unavailable = all_indices - set(available_neighbors_indices)
    probabilities_masked[list(unavailable)] = 0.0
    return probabilities_masked

def friendliness_probability(
    cluster: Set[Tuple[int, int]],
    friendliness: float,
    available_neighbors: List[Tuple[int, int]],
    available_neighbors_indices: List[int],
    neighborhood: Sequence[Tuple[int, int]],
    sharpness: float
) -> np.ndarray:
    """
    Calculate the probabilities of attaching to a given site based on the number of neighbors
    Friendliness controls preference for attaching to sites with more or less neighbors
    friendliness = 0: prefer sites with fewer neighbors
    friendliness = 0.5: no preference
    friendliness = 1: prefer sites with more neighbors
    """

    assert 0.0 <= friendliness <= 1.0

    # select the available neighbors
    neighbor_counts = np.zeros(len(VALID_NEIGHBORS))
    for neighbor, idx in zip(available_neighbors, available_neighbors_indices):
        neighbor_counts[idx] = count_neighbors(neighbor, cluster, neighborhood)

    if neighbor_counts.sum() == 0:
        return neighbor_counts
    
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

def attach_to_cluster_neighborhood(
    cluster: Set[Tuple[int, int]],
    growth_site: Tuple[int, int],
    growth_mode: float,
    friendliness: float,
    neighborhood: Sequence[Tuple[int, int]],
    sharpness: float,
    rng: random.Random 
) -> Optional[Tuple[int, int]]:
    """
    Attach to the cluster neighborhood with parameterized probabilities
    """
    available_neighbors_indices, available_neighbors = find_available_neighbors(cluster, growth_site)
    if len(available_neighbors) == 0:
        return None

    growth_mode_probabilities = growth_mode_probability(growth_mode)
    neighbor_probabilities = friendliness_probability(cluster, friendliness, available_neighbors, available_neighbors_indices, neighborhood, sharpness)

    combined_probabilities = growth_mode_probabilities + neighbor_probabilities

    attachment_probs = mask_probabilities(combined_probabilities, available_neighbors_indices)
    attachment_probs = normalize_probabilities(attachment_probs).tolist()

    # edge case where all probabilities are 0
    if sum(attachment_probs) == 0:
        return None

    # select a new site based on the probabilities
    selected_neighbor = rng.choices(VALID_NEIGHBORS, weights=attachment_probs, k=1)[0]
    new_site = (growth_site[0] + selected_neighbor[0], growth_site[1] + selected_neighbor[1])
    return new_site