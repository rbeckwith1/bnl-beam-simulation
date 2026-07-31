"""Static storyboard renderer for longitudinal phase-space simulations.

Creates a grid of panels sampled from the same snapshots used by
core/animation.py. The storyboard shares the animation's time unwrapping,
energy scaling, color scale, and separatrix calculation so that each panel
matches the corresponding animation frame.

Place this file at core/storyboard.py alongside core/animation.py.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from core.kinematics import T_rf_ns


# -------------------------------------------------------------------------
# Publication/report plotting defaults
# -------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
})


def render_storyboard(
    snapshots,
    time_init_for_color,
    a_t,
    a_E,
    separatrix,
    T_s_turns,
    out_path,
    n_panels=6,
    panel_indices=None,
    ncols=3,
    extra_info="",
    center_on_bunch=False,
    dpi=300,
    suptitle=None,
):
    """Render selected longitudinal phase-space snapshots as a static grid.

    Parameters
    ----------
    snapshots : dict
        Recorded simulation snapshots. Expected keys include ``turns``,
        ``times``, ``dEs``, and ``Vrf``.
    time_init_for_color : array-like
        Initial particle times used to color the particles consistently.
    a_t : float
        Time-coordinate scale used for the particle color normalization.
    a_E : float
        Energy-coordinate scale. Retained for compatibility with existing
        calls to this function.
    separatrix : object
        Object providing the ``separatrix_dE`` method.
    T_s_turns : float
        Synchrotron period in turns. Retained for compatibility with existing
        calls to this function.
    out_path : str
        Output image path.
    n_panels : int, optional
        Number of evenly spaced panels to select. The first and final recorded
        frames are included. Ignored when ``panel_indices`` is supplied.
    panel_indices : list[int] or None, optional
        Explicit snapshot indices to plot.
    ncols : int, optional
        Number of panels per row. For a report, three columns is recommended.
    extra_info : str, optional
        Additional simulation information displayed beneath the panels.
    center_on_bunch : bool, optional
        If True, use one fixed zoomed time window centered on the selected
        particle distributions.
    dpi : int, optional
        Output resolution.
    suptitle : str or None, optional
        Overall title displayed above the storyboard.
    """
    if len(snapshots["turns"]) == 0:
        print("Storyboard skipped: no snapshots recorded.")
        return

    n_frames = len(snapshots["turns"])

    # ---------------------------------------------------------------------
    # Select snapshots
    # ---------------------------------------------------------------------
    if panel_indices is None:
        n_panels = min(n_panels, n_frames)
        panel_indices = sorted(
            set(
                np.round(
                    np.linspace(0, n_frames - 1, n_panels)
                ).astype(int)
            )
        )

    n_panels = len(panel_indices)

    if n_panels == 0:
        print("Storyboard skipped: no panel indices selected.")
        return

    if ncols < 1:
        raise ValueError("ncols must be at least 1.")

    ncols = min(ncols, n_panels)

    # ---------------------------------------------------------------------
    # Establish common axis scales
    # ---------------------------------------------------------------------
    t_sep_array = np.linspace(-T_rf_ns / 2, T_rf_ns / 2, 400)

    margin = 1.0
    t_plot_lim = margin * (T_rf_ns / 2)

    Vrf_arr = np.asarray(snapshots["Vrf"])

    phi_s_arr = np.asarray(
        snapshots.get("phi_s", [0.0] * n_frames)
    )

    phi_ref_arr = np.asarray(
        snapshots.get(
            "phi_ref",
            [np.pi - phi_s for phi_s in phi_s_arr],
        )
    )

    i_max = int(np.argmax(Vrf_arr))
    phi_ref_max = phi_ref_arr[i_max]

    dE_pos_max, dE_neg_max = separatrix.separatrix_dE(
        t_sep_array,
        Vrf_arr[i_max],
        phi_ref=phi_ref_max,
    )

    all_sep_values = np.concatenate([
        np.asarray(dE_pos_max),
        np.asarray(dE_neg_max),
    ])

    finite_sep_values = all_sep_values[np.isfinite(all_sep_values)]

    if finite_sep_values.size == 0:
        raise ValueError(
            "Could not determine the energy-axis range from the separatrix."
        )

    dE_plot_lim_MeV = (
        margin * np.max(np.abs(finite_sep_values)) * 1e3
    )

    # ---------------------------------------------------------------------
    # Unwrap particle times exactly as in the animation
    # ---------------------------------------------------------------------
    t_plot_frames = []

    running_anchor = np.median(snapshots["times"][0])

    for i in range(n_frames):
        t_snap = np.asarray(snapshots["times"][i])

        t_plot = (
            running_anchor
            + (
                (
                    t_snap
                    - running_anchor
                    + T_rf_ns / 2.0
                )
                % T_rf_ns
            )
            - T_rf_ns / 2.0
        )

        t_plot_frames.append(t_plot)
        running_anchor = np.median(t_plot)

    # ---------------------------------------------------------------------
    # Optional fixed zoom around selected bunch distributions
    # ---------------------------------------------------------------------
    if center_on_bunch:
        pad = 1.5
        minimum_half_width_ns = 5.0

        all_t = np.concatenate(
            [t_plot_frames[i] for i in panel_indices]
        )

        t_min_all = np.min(all_t)
        t_max_all = np.max(all_t)

        t_center_fixed = 0.5 * (t_min_all + t_max_all)

        t_half_fixed = max(
            pad * 0.5 * (t_max_all - t_min_all),
            minimum_half_width_ns,
        )

        t_sep_frame_fixed = np.linspace(
            t_center_fixed - t_half_fixed,
            t_center_fixed + t_half_fixed,
            400,
        )

    # ---------------------------------------------------------------------
    # Figure layout
    # ---------------------------------------------------------------------
    nrows = int(np.ceil(n_panels / ncols))

    # These dimensions make each panel approximately 4.8 × 4.0 inches.
    # At 12 pt, the text remains legible after inclusion in a report.
    panel_width = 4.8
    panel_height = 4.0

    fig_width = panel_width * ncols + 1.1
    fig_height = panel_height * nrows

    if suptitle:
        fig_height += 0.45

    if extra_info:
        fig_height += 0.45

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    axes = axes.ravel()

    K0_list = snapshots.get("K0")
    K0_init = K0_list[0] if K0_list is not None else None

    scat = None

    # ---------------------------------------------------------------------
    # Draw each selected panel
    # ---------------------------------------------------------------------
    for panel_num, snapshot_index in enumerate(panel_indices):
        ax = axes[panel_num]

        t_plot = t_plot_frames[snapshot_index]
        dE_snap = np.asarray(snapshots["dEs"][snapshot_index])

        turn_snap = snapshots["turns"][snapshot_index]
        Vrf_snap = snapshots["Vrf"][snapshot_index]

        phi_s_snap = snapshots.get(
            "phi_s",
            [0.0] * n_frames,
        )[snapshot_index]

        phi_ref_snap = snapshots.get(
            "phi_ref",
            [np.pi - phi_s_snap] * n_frames,
        )[snapshot_index]

        scat = ax.scatter(
            t_plot,
            dE_snap * 1e3,
            c=time_init_for_color,
            cmap="twilight",
            s=7,
            vmin=-a_t,
            vmax=a_t,
            rasterized=True,
        )

        t_sep_frame = (
            t_sep_frame_fixed
            if center_on_bunch
            else t_sep_array
        )

        dE_pos, dE_neg = separatrix.separatrix_dE(
            t_sep_frame,
            Vrf_snap,
            phi_ref=phi_ref_snap,
            dE_search_max=separatrix.dE_grid_max,
        )

        ax.plot(
            t_sep_frame,
            np.asarray(dE_pos) * 1e3,
            "r-",
            linewidth=2.0,
        )

        ax.plot(
            t_sep_frame,
            np.asarray(dE_neg) * 1e3,
            "r-",
            linewidth=2.0,
        )

        if center_on_bunch:
            ax.set_xlim(
                t_center_fixed - t_half_fixed,
                t_center_fixed + t_half_fixed,
            )
        else:
            ax.set_xlim(-t_plot_lim, t_plot_lim)

        ax.set_ylim(
            -dE_plot_lim_MeV,
            dE_plot_lim_MeV,
        )

        ax.set_xlabel("Time deviation [ns]", fontsize=12)
        ax.set_ylabel("Energy deviation [MeV]", fontsize=12)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=12,
        )

        ax.grid(
            True,
            linewidth=0.5,
            alpha=0.25,
        )

        K0_line = ""

        if K0_list is not None:
            K0_snap = K0_list[snapshot_index]
            dK0_MeV = (K0_snap - K0_init) * 1e3

            K0_line = (
                f"\n$K_0$ = {K0_snap:.4f} GeV"
                f"\n$\\Delta K_0$ = {dK0_MeV:+.2f} MeV"
            )

        label = (
            f"Turn = {turn_snap:d}"
            f"\n$V_{{\\mathrm{{RF}}}}$ = "
            f"{Vrf_snap * 1e9 / 1e3:.1f} kV"
        )

        if not np.isclose(phi_s_snap, 0.0):
            label += (
                f"\n$\\phi_s$ = "
                f"{np.rad2deg(phi_s_snap):.1f}$^\\circ$"
            )

        label += K0_line

        ax.text(
            0.03,
            0.97,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.5",
                "alpha": 0.88,
            },
        )

    # ---------------------------------------------------------------------
    # Hide unused subplot cells
    # ---------------------------------------------------------------------
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    # ---------------------------------------------------------------------
    # Shared title, colorbar, and annotation
    # ---------------------------------------------------------------------
    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=14,
            y=0.985,
        )

    # Reserve space on the right for the shared colorbar.
    right_boundary = 0.88

    if extra_info:
        fig.text(
            0.46,
            0.015,
            extra_info,
            ha="center",
            va="bottom",
            fontsize=12,
            wrap=True,
        )
        bottom_boundary = 0.08
    else:
        bottom_boundary = 0.05

    top_boundary = 0.93 if suptitle else 0.97

    fig.subplots_adjust(
        left=0.08,
        right=right_boundary,
        bottom=bottom_boundary,
        top=top_boundary,
        wspace=0.30,
        hspace=0.32,
    )

    if scat is not None:
        cbar_ax = fig.add_axes([
            0.905,
            bottom_boundary + 0.03,
            0.018,
            top_boundary - bottom_boundary - 0.06,
        ])

        cbar = fig.colorbar(
            scat,
            cax=cbar_ax,
        )

        cbar.set_label(
            "Initial time [ns]",
            fontsize=12,
        )

        cbar.ax.tick_params(
            labelsize=12,
        )

    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    print(f"Storyboard saved to: {out_path}")

    plt.close(fig)