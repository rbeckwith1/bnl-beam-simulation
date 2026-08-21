"""Static storyboard renderer for longitudinal phase-space simulations.

Creates a grid of panels sampled from the same snapshots used by
core/animation.py. The storyboard shares the animation's time unwrapping,
energy scaling, and separatrix calculation so that each panel matches the
corresponding animation frame.

Design notes
------------
* Per-panel information is drawn as a title above each panel, never as an
  inset box, so the separatrix is never occluded.
* The default colormap is diverging, not cyclic. A cyclic map such as
  ``twilight`` assigns nearly the same color to the head and the tail of the
  bunch, which destroys the one thing the coloring is meant to show.
* Each panel reports the RMS bunch length, so the compression is legible
  even when the distribution is only a few millimetres across on the page.
* Axis labels are written once for the whole figure and are shortened
  automatically when the figure is too small to hold the full wording.
* The figure is built to an exact physical width (``fig_width_in``) and
  saved without a tight bounding box, so including it at 100% scale
  reproduces text at ``font_size_pt``. ``dpi`` changes pixel count only.

Place this file at core/storyboard.py alongside core/animation.py.
"""

import re

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from core.kinematics import T_rf_ns


# Rough width of one character as a fraction of the font size, used only for
# layout estimates, so an approximate value is adequate.
_CHAR_WIDTH_FRACTION = 0.58

# Vertical advance of one line of text, in units of the font size.
_LINE_HEIGHT_FRACTION = 1.30


def _visible_len(text):
    """Approximate the rendered character count of a mathtext string."""
    stripped = text.replace("$", "")

    # Each LaTeX command renders as roughly one glyph.
    stripped = re.sub(r"\\[A-Za-z]+", "x", stripped)

    # Braces, carets and underscores are markup, not glyphs.
    stripped = re.sub(r"[{}^_]", "", stripped)

    return len(stripped)


def _fit_label(candidates, available_in, em_in):
    """Return the first candidate label that fits within ``available_in``."""
    for text in candidates:
        width_in = _visible_len(text) * _CHAR_WIDTH_FRACTION * em_in

        if width_in <= available_in:
            return text

    return candidates[-1]


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
    fig_width_in=7.0,
    panel_aspect=0.85,
    font_size_pt=7,
    show_energy_lines=None,
    show_sigma_t=None,
    cmap="coolwarm",
    marker_size=1.0,
    marker_alpha=0.55,
    n_ticks=3,
):
    """Render selected longitudinal phase-space snapshots as a static grid.

    Parameters
    ----------
    snapshots : dict
        Recorded simulation snapshots. Expected keys include ``turns``,
        ``times``, ``dEs``, and ``Vrf``. If a ``sigma_t_ns`` key is present
        it is used directly; otherwise the RMS bunch length is computed from
        the unwrapped particle times.
    time_init_for_color : array-like
        Initial particle times used to color the particles consistently.
    a_t : float
        Time-coordinate scale used for the particle color normalization.
    a_E : float
        Energy-coordinate scale. Retained for compatibility.
    separatrix : object
        Object providing the ``separatrix_dE`` method.
    T_s_turns : float
        Synchrotron period in turns. Retained for compatibility.
    out_path : str
        Output image path.
    n_panels : int, optional
        Number of evenly spaced panels to select. Ignored when
        ``panel_indices`` is supplied. For a ramp whose voltage grows
        roughly exponentially, even spacing in turn number oversamples the
        beginning; prefer explicit ``panel_indices``.
    panel_indices : list[int] or None, optional
        Explicit snapshot indices to plot.
    ncols : int, optional
        Number of panels per row.
    extra_info : str, optional
        Deprecated and ignored. Put run information in the report caption.
    center_on_bunch : bool, optional
        If True, use one fixed zoomed time window centered on the selected
        particle distributions.
    dpi : int, optional
        Output resolution in pixels per inch. Does not affect layout.
    suptitle : str or None, optional
        Overall title. Leave as None for a report figure.
    fig_width_in : float, optional
        Total figure width in inches. Set to the destination text width and
        insert the image at 100% scale.
    panel_aspect : float, optional
        Panel height divided by panel width.
    font_size_pt : int, optional
        Size of every text element, in points. Margins scale with it.
    show_energy_lines : bool or None, optional
        Whether to print K_0 and Delta K_0. None decides automatically:
        shown only if the beam energy changes by more than 0.01 MeV.
    show_sigma_t : bool, optional
        Print the RMS bunch length in each panel title.
    cmap : str, optional
        Colormap for the initial-time coloring. Use a diverging map so the
        head and tail of the bunch are distinguishable.
    marker_size : float, optional
        Scatter marker area. Small semi-transparent markers show the shape
        of the distribution; large opaque ones show only a blob.
    marker_alpha : float, optional
        Scatter marker opacity.
    n_ticks : int, optional
        Approximate number of major ticks per axis.
    """
    if len(snapshots["turns"]) == 0:
        print("Storyboard skipped: no snapshots recorded.")
        return

    n_frames = len(snapshots["turns"])

    plt.rcParams.update({
        "font.size": font_size_pt,
        "axes.titlesize": font_size_pt,
        "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt,
        "ytick.labelsize": font_size_pt,
        "legend.fontsize": font_size_pt,
        "figure.titlesize": font_size_pt,
    })

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
    nrows = int(np.ceil(n_panels / ncols))

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
    # RMS bunch length per snapshot
    # ---------------------------------------------------------------------
    sigma_t_list = snapshots.get("sigma_t_ns")

    if sigma_t_list is None:
        sigma_t_list = [
            float(np.std(t_plot_frames[i]))
            for i in range(n_frames)
        ]

    # ---------------------------------------------------------------------
    # Decide whether the energy lines are worth the vertical space
    # ---------------------------------------------------------------------
    K0_list = snapshots.get("K0")
    K0_init = K0_list[0] if K0_list is not None else None

    if show_energy_lines is None:
        if K0_list is None:
            show_energy_lines = False
        else:
            dK0_MeV_max = max(
                abs((K0_list[i] - K0_init) * 1e3)
                for i in panel_indices
            )
            show_energy_lines = dK0_MeV_max > 0.01

    phi_s_all = snapshots.get("phi_s", [0.0] * n_frames)

    show_phi_s = any(
        not np.isclose(phi_s_all[i], 0.0)
        for i in panel_indices
    )

    n_title_lines = (
        2
        + (1 if show_sigma_t else 0)
        + (1 if show_phi_s else 0)
        + (2 if show_energy_lines else 0)
    )

    # ---------------------------------------------------------------------
    # Establish common axis scales
    # ---------------------------------------------------------------------
    t_sep_array = np.linspace(-T_rf_ns / 2, T_rf_ns / 2, 400)

    margin = 1.0
    t_plot_lim = margin * (T_rf_ns / 2)

    Vrf_arr = np.asarray(snapshots["Vrf"])

    phi_ref_arr = np.asarray(
        snapshots.get(
            "phi_ref",
            [np.pi - phi_s for phi_s in phi_s_all],
        )
    )

    i_max = int(np.argmax(Vrf_arr))

    dE_pos_max, dE_neg_max = separatrix.separatrix_dE(
        t_sep_array,
        Vrf_arr[i_max],
        phi_ref=phi_ref_arr[i_max],
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
    # Figure geometry
    # ---------------------------------------------------------------------
    em_in = font_size_pt / 72.0

    title_height_in = n_title_lines * _LINE_HEIGHT_FRACTION * em_in

    wspace = 0.10

    left_in = 0.30 + 2.8 * em_in     # shared y label + y tick labels
    right_in = 0.34 + 5.4 * em_in    # colorbar, its ticks and its label
    bottom_in = 0.18 + 2.6 * em_in   # shared x label + x tick labels
    top_in = (
        0.10
        + title_height_in
        + (1.9 * em_in if suptitle else 0.0)
    )

    axes_width_in = fig_width_in - left_in - right_in

    if axes_width_in <= 0:
        raise ValueError(
            f"fig_width_in={fig_width_in} in is too small at "
            f"{font_size_pt} pt: margins alone need "
            f"{left_in + right_in:.2f} in."
        )

    panel_width_in = axes_width_in / (ncols + wspace * (ncols - 1))
    panel_height_in = panel_aspect * panel_width_in

    # Rows must be separated by enough room for the titles.
    hspace = title_height_in / panel_height_in + 0.06

    axes_height_in = panel_height_in * (nrows + hspace * (nrows - 1))
    fig_height_in = axes_height_in + bottom_in + top_in

    # ---------------------------------------------------------------------
    # Fit checks
    # ---------------------------------------------------------------------
    longest_title_chars = 16
    min_panel_width_in = longest_title_chars * _CHAR_WIDTH_FRACTION * em_in

    if panel_width_in < min_panel_width_in:
        needed = (
            left_in
            + right_in
            + min_panel_width_in * (ncols + wspace * (ncols - 1))
        )
        print(
            f"WARNING: panels are {panel_width_in:.2f} in wide but the "
            f"titles need ~{min_panel_width_in:.2f} in at {font_size_pt} pt. "
            f"Use fig_width_in >= {needed:.1f}, fewer columns, or a smaller "
            f"font_size_pt."
        )

    # Axis labels are shortened rather than clipped when space runs out.
    y_label = _fit_label(
        [
            "Energy deviation [MeV]",
            "$\\Delta E$ [MeV]",
            "[MeV]",
        ],
        axes_height_in,
        em_in,
    )

    x_label = _fit_label(
        [
            "Time deviation [ns]",
            "$\\Delta t$ [ns]",
            "[ns]",
        ],
        axes_width_in,
        em_in,
    )

    cbar_label = _fit_label(
        [
            "Initial time [ns]",
            "$t_0$ [ns]",
            "[ns]",
        ],
        axes_height_in,
        em_in,
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width_in, fig_height_in),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

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

        phi_s_snap = phi_s_all[snapshot_index]
        phi_ref_snap = phi_ref_arr[snapshot_index]

        scat = ax.scatter(
            t_plot,
            dE_snap * 1e3,
            c=time_init_for_color,
            cmap=cmap,
            s=marker_size,
            alpha=marker_alpha,
            linewidths=0.0,
            vmin=-a_t,
            vmax=a_t,
            rasterized=True,
            zorder=2,
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

        for branch in (dE_pos, dE_neg):
            ax.plot(
                t_sep_frame,
                np.asarray(branch) * 1e3,
                "r-",
                linewidth=0.75,
                zorder=3,
            )

        if center_on_bunch:
            ax.set_xlim(
                t_center_fixed - t_half_fixed,
                t_center_fixed + t_half_fixed,
            )
        else:
            ax.set_xlim(-t_plot_lim, t_plot_lim)

        ax.set_ylim(-dE_plot_lim_MeV, dE_plot_lim_MeV)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=font_size_pt,
        )

        ax.grid(True, linewidth=0.5, alpha=0.25)

        # -----------------------------------------------------------------
        # Panel title: above the axes, so nothing is hidden
        # -----------------------------------------------------------------
        title_lines = [
            f"Turn = {turn_snap:d}",
            f"$V_{{\\mathrm{{RF}}}}$ = {Vrf_snap * 1e9 / 1e3:.1f} kV",
        ]

        if show_sigma_t:
            title_lines.append(
                f"$\\sigma_t$ = {sigma_t_list[snapshot_index]:.2f} ns"
            )

        if show_phi_s:
            title_lines.append(
                f"$\\phi_s$ = {np.rad2deg(phi_s_snap):.1f}$^\\circ$"
            )

        if show_energy_lines and K0_list is not None:
            K0_snap = K0_list[snapshot_index]
            dK0_MeV = (K0_snap - K0_init) * 1e3

            title_lines.append(f"$K_0$ = {K0_snap:.4f} GeV")
            title_lines.append(f"$\\Delta K_0$ = {dK0_MeV:+.2f} MeV")

        ax.set_title(
            "\n".join(title_lines),
            fontsize=font_size_pt,
            linespacing=1.15,
            pad=3.0,
        )

    # ---------------------------------------------------------------------
    # Hide unused cells and restore x tick labels where needed
    # ---------------------------------------------------------------------
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

        above = j - ncols

        if 0 <= above < n_panels:
            axes[above].tick_params(axis="x", labelbottom=True)

    # ---------------------------------------------------------------------
    # Shared labels, title and colorbar
    # ---------------------------------------------------------------------
    left_frac = left_in / fig_width_in
    right_frac = 1.0 - right_in / fig_width_in
    bottom_frac = bottom_in / fig_height_in
    top_frac = 1.0 - top_in / fig_height_in

    fig.subplots_adjust(
        left=left_frac,
        right=right_frac,
        bottom=bottom_frac,
        top=top_frac,
        wspace=wspace,
        hspace=hspace,
    )

    fig.text(
        0.5 * (left_frac + right_frac),
        0.30 * bottom_in / fig_height_in,
        x_label,
        ha="center",
        va="center",
        fontsize=font_size_pt,
    )

    fig.text(
        0.22 * left_in / fig_width_in,
        0.5 * (bottom_frac + top_frac),
        y_label,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=font_size_pt,
    )

    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=font_size_pt,
            y=1.0 - 0.30 * (1.9 * em_in) / fig_height_in,
        )

    if scat is not None:
        cbar_ax = fig.add_axes([
            right_frac + 0.28 * right_in / fig_width_in,
            bottom_frac,
            0.14 * right_in / fig_width_in,
            top_frac - bottom_frac,
        ])

        cbar = fig.colorbar(scat, cax=cbar_ax)

        cbar.locator = MaxNLocator(nbins=n_ticks + 1)
        cbar.update_ticks()

        cbar.set_label(cbar_label, fontsize=font_size_pt)
        cbar.ax.tick_params(labelsize=font_size_pt)
        cbar.solids.set_alpha(1.0)

    fig.savefig(out_path, dpi=dpi)

    sigma_first = sigma_t_list[panel_indices[0]]
    sigma_last = sigma_t_list[panel_indices[-1]]

    print(
        f"Storyboard saved to: {out_path} | "
        f"{fig_width_in:.2f} x {fig_height_in:.2f} in | "
        f"{nrows} x {ncols} grid | "
        f"panels {panel_width_in:.2f} x {panel_height_in:.2f} in | "
        f"{font_size_pt} pt | "
        f"sigma_t {sigma_first:.2f} -> {sigma_last:.2f} ns"
    )

    plt.close(fig)