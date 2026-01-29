import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _prepare_scatter(ax, point_size=0.6):
    ax.set_aspect("equal", "box")
    ax.axis("off")
    sc = ax.scatter([], [], s=point_size, c=[], cmap="cool", vmin=0, vmax=1)
    return sc

def _set_scatter(sc, cluster_history):
    xs = [x for (x, _) in cluster_history]
    ys = [y for (_, y) in cluster_history]
    sc.set_offsets(np.column_stack([xs, ys]))
    history = np.linspace(0, 1, len(cluster_history))
    sc.set_array(history)


def make_param_sweep_animation(
    parameter_name: str,
    parameter_values,
    dla_simulation,
    sim_defaults: dict,
    plot_defaults: dict,
    fixed_overrides: dict | None = None,
    interval_ms: int = 250,
):
    """
    1) Precompute cluster_history voor elke param value
    2) Animeer de scatter per param value
    """

    fixed_overrides = fixed_overrides or {}

    # --- precompute ---
    histories = []
    for v in parameter_values:
        kwargs = dict(sim_defaults)
        kwargs.update(fixed_overrides)
        kwargs[parameter_name] = float(v)

        # simulate_dla returns: cluster, cluster_history, origin, mass_history, max_r2
        _, cluster_history, _, _, _ = dla_simulation(**kwargs)
        histories.append(cluster_history)

    # --- set up figure ---
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = _prepare_scatter(ax, point_size=plot_defaults.get("point_size", 0.6))

    # Optional: vaste limieten zodat plot niet “springt”
    # (bepaal globale min/max over alle frames)
    all_x = [x for h in histories for (x, _) in h]
    all_y = [y for h in histories for (_, y) in h]
    pad = 2
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    title_prefix = plot_defaults.get("title_cluster", "Parameterized DLA")
    title_text = ax.set_title("")

    def init():
        _set_scatter(sc, histories[0])
        title_text.set_text(f"{title_prefix}\n{parameter_name}={parameter_values[0]:.3f}")
        return sc, title_text


    def update(i):
        _set_scatter(sc, histories[i])
        title_text.set_text(f"{title_prefix}\n{parameter_name}={parameter_values[i]:.3f}")
        return sc, title_text

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
