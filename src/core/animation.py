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
                      T_s_turns, out_path, fps=30, extra_info="",
                      center_on_bunch=False):
    if len(snapshots["turns"]) == 0:
        print("Animation skipped: no snapshots recorded.")
        return

    t_sep_array = np.linspace(-T_rf_ns / 2, T_rf_ns / 2, 200)
    margin = 1
    t_plot_lim = margin * (T_rf_ns / 2)

    Vrf_arr = np.asarray(snapshots["Vrf"])
    phi_s_arr = np.asarray(snapshots.get("phi_s", [0.0] * len(snapshots["turns"])))
    phi_ref_arr = np.asarray(snapshots.get("phi_ref", [np.pi - p for p in phi_s_arr]))
    i_max = np.argmax(Vrf_arr)
    phi_ref_max = phi_ref_arr[i_max]
    dE_pos_max, dE_neg_max = separatrix.separatrix_dE(
        t_sep_array, Vrf_arr[i_max], phi_ref=phi_ref_max
    )
    dE_plot_lim_MeV = margin * np.nanmax(
        np.abs(np.concatenate([dE_pos_max, dE_neg_max]))
    ) * 1e3

    n_frames = len(snapshots["turns"])

    # --- pre-unwrap every frame's t-coordinate once, up front, so a bunch
    # straddling the periodic wrap boundary (t_ns has period T_rf_ns)
    # appears as one contiguous group rather than split at the two edges.
    # Each frame is unwrapped relative to the previous frame's center, so
    # this tracks the bunch smoothly across turns even as it drifts.
    t_plot_frames = []
    running_anchor = np.median(snapshots["times"][0])
    for i in range(n_frames):
        t_snap = snapshots["times"][i]
        t_plot = running_anchor + ((t_snap - running_anchor + T_rf_ns / 2.0) % T_rf_ns) - T_rf_ns / 2.0
        t_plot_frames.append(t_plot)
        running_anchor = np.median(t_plot)

    if center_on_bunch:
        # one-time bounding box over the WHOLE run -> stationary window
        pad = 1.5
        floor_t_ns = 5.0
        all_t = np.concatenate(t_plot_frames)
        t_min_all, t_max_all = np.min(all_t), np.max(all_t)
        t_center_fixed = 0.5 * (t_min_all + t_max_all)
        t_half_fixed = max(pad * 0.5 * (t_max_all - t_min_all), floor_t_ns)
        t_sep_frame_fixed = np.linspace(
            t_center_fixed - t_half_fixed, t_center_fixed + t_half_fixed, 200
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    if center_on_bunch:
        ax.set_xlim(t_center_fixed - t_half_fixed, t_center_fixed + t_half_fixed)
    else:
        ax.set_xlim(-t_plot_lim, t_plot_lim)
    ax.set_ylim(-dE_plot_lim_MeV, dE_plot_lim_MeV)

    ax.set_xlabel("Time deviation [ns]")
    ax.set_ylabel("Energy deviation [MeV]")
    scat = ax.scatter([], [], c=[], cmap="twilight", s=3, vmin=-a_t, vmax=a_t)
    sep_pos_line, = ax.plot([], [], "r-", lw=1.5)
    sep_neg_line, = ax.plot([], [], "r-", lw=1.5)
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                         fontsize=9, family="monospace",
                         bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    K0_list = snapshots.get("K0", None)
    K0_init = K0_list[0] if K0_list else None

    def init_anim():
        scat.set_offsets(np.empty((0, 2)))
        sep_pos_line.set_data([], [])
        sep_neg_line.set_data([], [])
        info_text.set_text("")
        return scat, sep_pos_line, sep_neg_line, info_text

    def update_anim(i):
        t_plot = t_plot_frames[i]
        dE_snap = snapshots["dEs"][i]
        turn_snap = snapshots["turns"][i]
        Vrf_snap = snapshots["Vrf"][i]
        phi_s_snap = snapshots.get("phi_s", [0.0] * n_frames)[i]
        phi_ref_snap = snapshots.get("phi_ref", [np.pi - phi_s_snap] * n_frames)[i]

        scat.set_offsets(np.column_stack([t_plot, dE_snap * 1e3]))
        scat.set_array(time_init_for_color)

        if center_on_bunch:
            t_sep_frame = t_sep_frame_fixed   # window is fixed -- set once, outside the loop
        else:
            t_sep_frame = t_sep_array

        dE_pos, dE_neg = separatrix.separatrix_dE(
            t_sep_frame, Vrf_snap, phi_ref=phi_ref_snap,
            dE_search_max=separatrix.dE_grid_max,
        )
        sep_pos_line.set_data(t_sep_frame, dE_pos * 1e3)
        sep_neg_line.set_data(t_sep_frame, dE_neg * 1e3)

        K0_line = ""
        if K0_list is not None:
            K0_snap = K0_list[i]
            dK0_MeV = (K0_snap - K0_init) * 1e3
            K0_line = f"\nK0 = {K0_snap:.6f} GeV (+{dK0_MeV:.3f} MeV)"

        info_text.set_text(
            f"turn = {turn_snap:d}\nVrf = {Vrf_snap*1e9/1e3:.1f} kV\n"
            f"T_s (ref) = {T_s_turns:.1f} turns"
            + (f"\nphi_s = {np.rad2deg(phi_s_snap):.1f} deg" if phi_s_snap != 0.0 else "")
            + K0_line
            + (f"\n{extra_info}" if extra_info else "")
        )
        return scat, sep_pos_line, sep_neg_line, info_text

    anim = animation.FuncAnimation(
        fig, update_anim, frames=n_frames, init_func=init_anim,
        blit=False, interval=1000 / fps
    )
    try:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
        print(f"Animation saved to: {out_path}")
    except Exception as exc:
        warnings.warn(f"Could not save animation (is ffmpeg installed?): {exc}")
    plt.close(fig)