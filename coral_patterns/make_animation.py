import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def prepare_scatter(ax, point_size=0.6):
    """
    initialize empty scatter plot, such that it can be used to plot
    the cluster history in for the gif. Returns an empty scatter
    """
    ax.set_aspect("equal", "box")
    ax.axis("off")
    sc = ax.scatter([], [], s=point_size, c=[], cmap="cool", vmin=0, vmax=1)
    return sc

def set_scatter(sc, cluster_history):
    """
    This function makes a scatter of the produced clusters, s.t. the 
    empty scatter is filled with data.
    """
    xs = [x for (x, _) in cluster_history]
    ys = [y for (_, y) in cluster_history]
    sc.set_offsets(np.column_stack([xs, ys]))
    history = np.linspace(0, 1, len(cluster_history))
    sc.set_array(history)


def make_param_sweep_animation(parameter_name, parameter_values, dla_simulation, sim_defaults, plot_defaults,
    fixed_overrides, interval_ms= 250):
    """
    This function performs:
    1) The actual parametersweep over the desired parameter space.
    2) Makes the scatter of the coral cluster
    """
    fixed_overrides = fixed_overrides or {}

    # setup parameter environment for the animation
    histories = []
    for v in parameter_values:
        kwargs = dict(sim_defaults)
        kwargs.update(fixed_overrides)
        kwargs[parameter_name] = float(v)

        # simulate the dla and store the cluster history as the results of
        # the simulations, which will be displayed in the final gif
        _, cluster_history, _, _, _ = dla_simulation(**kwargs)
        histories.append(cluster_history)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = prepare_scatter(ax, point_size=plot_defaults.get("point_size", 0.6))

    
    all_x = [x for h in histories for (x, _) in h]
    all_y = [y for h in histories for (_, y) in h]
    pad = 2
    # Global limits set, such that the gif does not jump while displaying
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    title_prefix = plot_defaults.get("title_cluster", "Parameterized DLA")
    title_text = ax.set_title("")

    def init():
        """
        Initialize the actual scatter.
        """
        set_scatter(sc, histories[0])
        title_text.set_text(f"{title_prefix}\n{parameter_name}={parameter_values[0]:.3f}")
        return sc, title_text


    def update(i):
        """
        This function updates every prior scatter, 
        such that a new frame is added to it.
        """
        set_scatter(sc, histories[i])
        title_text.set_text(f"{title_prefix}\n{parameter_name}={parameter_values[i]:.3f}")
        return sc, title_text

    # Call the function animation and build it
    anim = FuncAnimation(
        fig,
        update,
        frames=len(parameter_values),
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )
    return anim
