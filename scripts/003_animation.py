import numpy as np 
from coral_patterns.make_animation import make_param_sweep_animation
from coral_patterns.Simulation import simulate_dla
from coral_patterns.config import DEFAULTS, PLOT_DEFAULTS

growth_values = np.linspace(-1, 1, 21)

# growth_mode parameter sweep
anim_growth = make_param_sweep_animation( 
    parameter_name="growth_mode",
    parameter_values=growth_values,
    dla_simulation=simulate_dla,
    sim_defaults=DEFAULTS,
    plot_defaults=PLOT_DEFAULTS,
    # fixed_overrides={}  # eventueel friendliness fixen etc.
    interval_ms=300,
)

# In Jupyter:
from IPython.display import HTML
HTML(anim_growth.to_jshtml())

# friendliness parameter sweep
friend_values = np.linspace(0, 1, 21)

anim_friend = make_param_sweep_animation(
    parameter_name="friendliness",
    parameter_values=friend_values,
    dla_simulation=simulate_dla,
    sim_defaults=DEFAULTS,
    plot_defaults=PLOT_DEFAULTS,
    # Bijvoorbeeld: growth_mode vastzetten:
    fixed_overrides={"growth_mode": DEFAULTS.get("growth_mode", 0)},
    interval_ms=300,
)

from IPython.display import HTML
HTML(anim_friend.to_jshtml())


anim_growth.save("growth_mode_sweep.gif", writer="pillow", fps=1)
anim_friend.save("friendliness_sweep.gif", writer="pillow", fps=1)