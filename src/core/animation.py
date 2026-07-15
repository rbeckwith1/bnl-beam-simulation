"""Shared phase-space animation renderer, driven by a Separatrix instance
(see core.separatrix) so it works identically for all three methods."""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from core.kinematics import T_rf_ns


def render_animation(snapshots, time_init_for_color, a_t, a_E, separatrix,
                      T_s_turns, out_path, fps=30, extra_info=""):
    if len(snapshots["turns"]) == 0:
        print("Animation skipped: no snapshots recorded.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    t_plot_lim = 20.0 * a_t
    dE_plot_lim_MeV = 20.0 * a_E * 1e3
    ax.set_xlim(-t_plot_lim , t_plot_lim)
    ax.set_ylim(-dE_plot_lim_MeV / 5, dE_plot_lim_MeV / 5)
    ax.set_xlabel("Time deviation [ns]")
    ax.set_ylabel("Energy deviation [MeV]")

    scat = ax.scatter([], [], c=[], cmap="twilight", s=3, vmin=-a_t, vmax=a_t)
    sep_pos_line, = ax.plot([], [], "r-", lw=1.5)
    sep_neg_line, = ax.plot([], [], "r-", lw=1.5)
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                         fontsize=9, family="monospace",
                         bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    t_sep_array = np.linspace(-T_rf_ns / 2, T_rf_ns / 2, 200)

    def init_anim():
        scat.set_offsets(np.empty((0, 2)))
        sep_pos_line.set_data([], [])
        sep_neg_line.set_data([], [])
        info_text.set_text("")
        return scat, sep_pos_line, sep_neg_line, info_text

    def update_anim(i):
        t_snap = snapshots["times"][i]
        dE_snap = snapshots["dEs"][i]
        turn_snap = snapshots["turns"][i]
        Vrf_snap = snapshots["Vrf"][i]

        scat.set_offsets(np.column_stack([t_snap, dE_snap * 1e3]))
        scat.set_array(time_init_for_color)

        dE_pos, dE_neg = separatrix.separatrix_dE(t_sep_array, Vrf_snap)
        sep_pos_line.set_data(t_sep_array, dE_pos * 1e3)
        sep_neg_line.set_data(t_sep_array, dE_neg * 1e3)

        info_text.set_text(
            f"turn = {turn_snap:d}\nVrf = {Vrf_snap*1e9/1e3:.1f} kV\n"
            f"T_s (ref) = {T_s_turns:.1f} turns\n{extra_info}"
        )
        return scat, sep_pos_line, sep_neg_line, info_text

    anim = animation.FuncAnimation(
        fig, update_anim, frames=len(snapshots["turns"]), init_func=init_anim,
        blit=False, interval=1000 / fps
    )
    try:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
        print(f"Animation saved to: {out_path}")
    except Exception as exc:
        warnings.warn(f"Could not save animation (is ffmpeg installed?): {exc}")
    plt.close(fig)
