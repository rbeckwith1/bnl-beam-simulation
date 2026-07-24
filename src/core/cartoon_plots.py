"""Static cartoon renderer for a poster: a small grid of
panels sampled from the same snapshots used by core/animation.py, sharing
axes, color scale, and separatrix so the sequence reads like film-strip
frames of the animation rather than a separate figure.

Reuses the exact same frame-prep logic as render_animation (t-unwrapping,
dE scaling) so a panel here matches what you'd see at that turn in the
.mp4.

Place this file at core/storyboard.py alongside core/animation.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.kinematics import T_rf_ns


def render_storyboard(snapshots, time_init_for_color, a_t, a_E, separatrix,
                       T_s_turns, out_path, n_panels=6, panel_indices=None,
                       ncols=3, extra_info="", center_on_bunch=False,
                       dpi=300, suptitle=None):
    """
    Parameters
    ----------
    n_panels : int
        Number of evenly-spaced panels to pick across the recorded
        snapshots (first and last frame are always included). Ignored if
        panel_indices is given.
    panel_indices : list[int] or None
        Explicit snapshot indices to plot instead of evenly-spaced picks
        (e.g. if you want to highlight a specific compression point).
    ncols : int
        Panels per row.
    dpi : int
        Poster-quality output resolution.
    """
    if len(snapshots["turns"]) == 0:
        print("Storyboard skipped: no snapshots recorded.")
        return

    n_frames = len(snapshots["turns"])

    # --- pick which snapshots become panels -------------------------------
    if panel_indices is None:
        n_panels = min(n_panels, n_frames)
        panel_indices = sorted(set(
            np.round(np.linspace(0, n_frames - 1, n_panels)).astype(int)
        ))
    n_panels = len(panel_indices)

    # --- same axis-scale setup as render_animation -------------------------
    t_sep_array = np.linspace(-T_rf_ns / 2, T_rf_ns / 2, 200)
    margin = 1
    t_plot_lim = margin * (T_rf_ns / 2)

    Vrf_arr = np.asarray(snapshots["Vrf"])
    phi_s_arr = np.asarray(snapshots.get("phi_s", [0.0] * n_frames))
    phi_ref_arr = np.asarray(snapshots.get("phi_ref", [np.pi - p for p in phi_s_arr]))
    i_max = np.argmax(Vrf_arr)
    phi_ref_max = phi_ref_arr[i_max]
    dE_pos_max, dE_neg_max = separatrix.separatrix_dE(
        t_sep_array, Vrf_arr[i_max], phi_ref=phi_ref_max
    )
    dE_plot_lim_MeV = margin * np.nanmax(
        np.abs(np.concatenate([dE_pos_max, dE_neg_max]))
    ) * 1e3

    # --- same t-unwrapping as render_animation, run over ALL frames so the
    # anchor tracking is identical even though we only plot a subset -------
    t_plot_frames = []
    running_anchor = np.median(snapshots["times"][0])
    for i in range(n_frames):
        t_snap = snapshots["times"][i]
        t_plot = running_anchor + ((t_snap - running_anchor + T_rf_ns / 2.0) % T_rf_ns) - T_rf_ns / 2.0
        t_plot_frames.append(t_plot)
        running_anchor = np.median(t_plot)

    if center_on_bunch:
        pad = 1.5
        floor_t_ns = 5.0
        all_t = np.concatenate([t_plot_frames[i] for i in panel_indices])
        t_min_all, t_max_all = np.min(all_t), np.max(all_t)
        t_center_fixed = 0.5 * (t_min_all + t_max_all)
        t_half_fixed = max(pad * 0.5 * (t_max_all - t_min_all), floor_t_ns)
        t_sep_frame_fixed = np.linspace(
            t_center_fixed - t_half_fixed, t_center_fixed + t_half_fixed, 200
        )

    # --- layout -------------------------------------------------------------
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    K0_list = snapshots.get("K0", None)
    K0_init = K0_list[0] if K0_list else None

    scat = None
    for panel_num, i in enumerate(panel_indices):
        ax = axes[panel_num]

        t_plot = t_plot_frames[i]
        dE_snap = snapshots["dEs"][i]
        turn_snap = snapshots["turns"][i]
        Vrf_snap = snapshots["Vrf"][i]
        phi_s_snap = snapshots.get("phi_s", [0.0] * n_frames)[i]
        phi_ref_snap = snapshots.get("phi_ref", [np.pi - phi_s_snap] * n_frames)[i]

        scat = ax.scatter(
            t_plot, dE_snap * 1e3, c=time_init_for_color,
            cmap="twilight", s=3, vmin=-a_t, vmax=a_t
        )

        t_sep_frame = t_sep_frame_fixed if center_on_bunch else t_sep_array
        dE_pos, dE_neg = separatrix.separatrix_dE(
            t_sep_frame, Vrf_snap, phi_ref=phi_ref_snap,
            dE_search_max=separatrix.dE_grid_max,
        )
        ax.plot(t_sep_frame, dE_pos * 1e3, "r-", lw=1.5)
        ax.plot(t_sep_frame, dE_neg * 1e3, "r-", lw=1.5)

        if center_on_bunch:
            ax.set_xlim(t_center_fixed - t_half_fixed, t_center_fixed + t_half_fixed)
        else:
            ax.set_xlim(-t_plot_lim, t_plot_lim)
        ax.set_ylim(-dE_plot_lim_MeV, dE_plot_lim_MeV)

        K0_line = ""
        if K0_list is not None:
            K0_snap = K0_list[i]
            dK0_MeV = (K0_snap - K0_init) * 1e3
            K0_line = f"\nK0={K0_snap:.4f} GeV (+{dK0_MeV:.2f} MeV)"

        label = (
            f"turn = {turn_snap:d}\nVrf = {Vrf_snap*1e9/1e3:.1f} kV"
            + (f"\nphi_s = {np.rad2deg(phi_s_snap):.1f} deg" if phi_s_snap != 0.0 else "")
            + K0_line
        )
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        if panel_num % ncols == 0:
            ax.set_ylabel("Energy deviation [MeV]")
        if panel_num >= n_panels - ncols:
            ax.set_xlabel("Time deviation [ns]")

    # hide any unused axes (when n_panels doesn't fill the grid)
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    if extra_info:
        fig.text(0.5, 0.005, extra_info, ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=[0, 0.03, 0.90, 0.96] if suptitle or extra_info else [0, 0, 0.90, 1])

    # single shared colorbar for the whole figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scat, cax=cbar_ax, label="Initial time [ns]")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Storyboard saved to: {out_path}")
    plt.close(fig)