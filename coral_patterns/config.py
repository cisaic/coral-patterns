DEFAULTS = {
    "target_mass": 10000,
    "launch_margin": 6,
    "kill_margin": 25,
    "max_steps_per_walker": 80_000,
    "rng_seed": 1,
    "log_every": 200,
    "growth_mode": 0, # defines whether to grow upward or downward 
    "friendliness": 0.5, # sparsity preference measure 
    "neighborhood_radius": 1,
    "sharpness": 1,
}

PLOT_DEFAULTS = {
    "point_size": 0.6,
    "title_cluster": "Parameterized DLA (8-neighborhood) — N=1000",
    "title_mass_radius": "Mass-radius scaling (log-log)",
    "title_mass_time": "Baseline growth: M(t)",
}


NEIGHBORS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
NEIGHBORS8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
VALID_NEIGHBORS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0),] 
