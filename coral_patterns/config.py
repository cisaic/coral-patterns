# config.py
DEFAULTS = {
    "target_mass": 4000,
    "launch_margin": 6,
    "kill_margin": 25,
    "max_steps_per_walker": 80_000,
    "rng_seed": 1,
    "log_every": 2000,
}

NEIGHBORS8 = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]

PAPER_PARAMS = {
    "smin": 0.0,
    "smax": 0.99,
    "nu_mm_per_year": 5.0,
    "delta_sub_mm": 5.0,
    "l_br_mm": 10.0,
    "theta_deg": 55.0,
}
