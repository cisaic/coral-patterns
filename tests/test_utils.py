import pytest
import mock
import numpy as np
import math
import random

from coral_patterns.helpers import normalize_probabilities, logspace, fit_line, generate_neighborhood, count_neighbors, mass_in_radius, sqdist, sqdist_point, radius_from_r2
from coral_patterns.dla import mask_probabilities, growth_mode_probability, friendliness_probability, attach_to_cluster_neighborhood, run_walker
from coral_patterns.config import NEIGHBORS8


def test_sqdist():
    x = 3
    y = 4
    expected = 25
    result = sqdist(x, y,)
    assert expected == result


def test_sqdist_point():
    p = (3, 9)
    origin = (0, 0)
    expected = 90
    result = sqdist_point(p, origin)
    assert expected == result


def test_radius_from_r2():
    assert radius_from_r2(0) == 0
    assert radius_from_r2(4) == 2
    assert radius_from_r2(8) == 3


def test_logspace_n():
    # test for n > 2
    a = 1
    b = math.exp(3)
    n = 4
    expected = np.array([1, np.exp(1), np.exp(2), np.exp(3)])
    result = np.array(logspace(a, b, n))
    assert np.allclose(result, expected)

    # test for n < 2
    a = 1
    b = 2
    n = 1
    expected = [a]
    result = logspace(a, b, n)
    assert np.allclose(result, expected)

    # errors
    with pytest.raises(ValueError):
        logspace(-1, 1, 4)

    with pytest.raises(ValueError):
        logspace(0, 1, 4)


def test_fit_line():
    xs = np.array([1, 1, 4])
    ys = np.array([2, 4, 6])
    expected = np.array([1.0, 2.0])
    result = np.array(fit_line(xs, ys))
    print(expected)
    print(result)
    assert np.allclose(result, expected)

    # test for den = 0
    xs = np.array([1, 1, 1])
    ys = np.array([2, 4, 6])
    m, _ = fit_line(xs, ys)
    assert np.isnan(m)


def test_normalize_probabilities():
    input_data = np.array([1, 2, 3, 4])
    expected = np.array([0.1, 0.2, 0.3, 0.4])
    result = normalize_probabilities(input_data)
    assert np.allclose(result, expected)


def test_generate_neighborhood():
    radius = 1
    expected = np.array([
        (-1, -1), 
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1), 
        (1, 0),
        (1, 1)
        ])
    result = np.array(generate_neighborhood(radius))
    assert np.allclose(result, expected)


def test_count_neighbors():
    site = (1, 2)
    cluster = [(0, 3), (0, 2), (0, 1), (1, 1)]
    neighborhood = [
        (-1, -1), 
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1), 
        (1, 0),
        (1, 1)
        ]
    expected = 4
    result = count_neighbors(site, cluster, neighborhood)
    assert expected == result

def test_mass_in_radius():
    cluster = [(0,2), (1, 1), (1, 0)]
    origin = (1, 2)
    r = 1
    expected = 2
    result = mass_in_radius(cluster, origin, r)
    assert expected == result


def test_run_walker():
    cluster = {(0, 0)}
    origin = (0, 0)
    launch_radius = 0.5
    kill_radius = 1.0
    max_steps = 10
    rng = random.Random(7)

    # set launch point at (1,0) and step (-1,0) to origin
    with mock.patch('coral_patterns.dla.launch_point', return_value=(1, 0)):
        with mock.patch('coral_patterns.dla.random_step', return_value=(-1, 0)):
            new_site, path = run_walker(
                cluster,
                origin=origin,
                launch_radius=launch_radius,
                kill_radius=kill_radius,
                max_steps=max_steps,
                rng=rng,
                record_path=True
            )

    # should stick to cluster
    assert new_site == (0, 0)

    # at least one step
    assert len(path) > 1


def test_mask_probabilities():
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    available_neighbors_indices = np.array([1, 2, 3])
    expected = np.array([0.0, 0.2, 0.3, 0.4, 0.0])
    result = np.array([mask_probabilities(probabilities, available_neighbors_indices)])
    assert np.allclose(result, expected)


def test_growth_mode_probability():
    growth_exp_prob = [(-1.0, [1.0, 0.0, 0.0, 0.0, 1.0]),
                       (-0.75, [1.0, 0.5, 0.0, 0.5, 1.0]),
                       (-0.5, [1.0, 1.0, 0.0, 1.0, 1.0]),
                       (-0.25, [1.0, 1.0, 0.5, 1.0, 1.0]),
                       (0.0, [1.0, 1.0, 1.0, 1.0, 1.0]),
                       (0.25, [0.5, 1.0, 1.0, 1.0, 0.5]),
                       (0.5, [0.0, 1.0, 1.0, 1.0, 0.0]),
                       (0.75, [0.0, 0.5, 1.0, 0.5, 0.0]),
                       (1.0, [0.0, 0.0, 1.0, 0.0, 0.0])]
    
    for g, expected in growth_exp_prob:
        result = np.array(growth_mode_probability(g))
        expected_norm = normalize_probabilities(np.array(expected))
        assert np.allclose(result, expected_norm)


def test_friendliness_probability():
    cluster = {(0, 0), (0, 1), (1, 0)}
    available_neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0)]
    available_neighbors_indices = [0, 1, 2, 3, 4]
    neighborhood = NEIGHBORS8
    sharpness = 1.0

    counts = [count_neighbors(n, cluster, neighborhood) for n in available_neighbors]

    # friendliness = 0 --> site with fewest neighbors should have highest probability
    probs_low = friendliness_probability(cluster, 0.0, available_neighbors, available_neighbors_indices, neighborhood, sharpness)
    min_count_idx = np.argmin(counts)
    assert np.argmax(probs_low) == min_count_idx

    # friendliness = 1 --> site with most neighbors should have highest probability
    probs_high = friendliness_probability(cluster, 1.0, available_neighbors, available_neighbors_indices, neighborhood, sharpness)
    max_count_idx = np.argmax(counts)
    assert np.argmax(probs_high) == max_count_idx

    # friendliness = 0.5 --> all availbal neighbors should have prob > 0.0
    probs_neutral = friendliness_probability(cluster, 0.5, available_neighbors, available_neighbors_indices, neighborhood, sharpness)
    assert np.all(probs_neutral > 0.0)


def test_attach_to_cluster_neighborhood():
    cluster = {(0, 0), (0, 1), (1, 0)}
    growth_site = (0, 0)
    growth_mode = 0.0
    friendliness = 0.5
    sharpness = 1.0
    neighborhood = NEIGHBORS8
    rng = random.Random(7)

    # choose (1,0)
    with mock.patch('random.Random.choices') as mock_choices:
        mock_choices.return_value = [(1, 0)]

        new_site = attach_to_cluster_neighborhood(
            cluster,
            growth_site,
            growth_mode,
            friendliness,
            neighborhood,
            sharpness,
            rng
        )

    assert new_site == (growth_site[0] + 1, growth_site[1] + 0)
