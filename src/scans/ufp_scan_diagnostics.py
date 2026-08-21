"""
UFP release-and-capture: dwell_turns scan diagnostics.

Reads results/ufp_scan/scan_summary.csv (built up by rerunning ufp_scan.py
with different DWELL_TURNS values) and produces the same family of plots you
used for the resonant mod_depth scan:

  1. overlay of all sigma_t(turn) traces, one per dwell_turns
  2. min sigma_t vs dwell_turns (color = turn at which the min occurred)
  3. two-panel: min sigma_t vs dwell_turns, and turn_of_min vs dwell_turns
  4. 2D heatmap: dwell_turns vs turn, colored by sigma_t, with the
     release/capture timing overlaid

This script assumes PHASE_JUMP_DEG and JUMP_TURNS were held fixed across the
scan (only DWELL_TURNS was varied). If you later sweep those too, add them to
the dedup/group-by key and this script will need a slice/filter step first.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from matplotlib.collections import LineCollection


# ---- paths ----
SCANS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.dirname(SCANS_DIR)
ROOT_DIR  = os.path.join(SRC_DIR, "results", "ufp_scan")
HIST_DIR  = os.path.join(ROOT_DIR, "histories")
SUMMARY_PATH = os.path.join(ROOT_DIR, "scan_summary.csv")
DIAG_DIR = os.path.join(SRC_DIR, "results", "ufp_scan_diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)


def plot_min_sigma_vs_dwell_turns(dwell_turns, min_sigmas, turns_of_min,
                                   highlight_range=None, save_path=None):
    """
    Pure plotting function -- knows nothing about files or paths.
    dwell_turns, min_sigmas, turns_of_min : 1D arrays, same length, sorted by dwell_turns
    highlight_range : optional (lo, hi) tuple to shade a validated smooth segment
    """
    dwell_turns = np.asarray(dwell_turns)
    min_sigmas = np.asarray(min_sigmas)
    turns_of_min = np.asarray(turns_of_min)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if highlight_range is not None:
        ax.axvspan(*highlight_range, color='gray', alpha=0.12, zorder=0)
    points = np.array([dwell_turns, min_sigmas]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', zorder=1)
    lc.set_array(turns_of_min[:-1])
    ax.add_collection(lc)
    sc = ax.scatter(dwell_turns, min_sigmas, c=turns_of_min, cmap='viridis',
                     s=45, edgecolor='k', linewidth=0.4, zorder=2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('turn_of_min')
    ax.set_xlabel('dwell_turns')
    ax.set_ylabel('min $\\sigma_t$ [ns]')
    ax.set_title('UFP capture: min $\\sigma_t$ vs dwell_turns\n(color = turn at which the minimum occurred)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig, ax


if __name__ == "__main__":
    summary = pd.read_csv(SUMMARY_PATH)
    summary = summary.drop_duplicates(
        subset=["dwell_turns", "phase_jump_deg", "jump_turns"], keep="last"
    )
    summary_sorted = summary.sort_values("dwell_turns")

    if summary_sorted["phase_jump_deg"].nunique() > 1 or summary_sorted["jump_turns"].nunique() > 1:
        print("Warning: phase_jump_deg or jump_turns varies across rows -- "
              "these plots assume a pure dwell_turns sweep. Filter summary "
              "first if you've mixed scans.")

    # ---- 1. overlay of all traces ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in summary_sorted.iterrows():
        hist = pd.read_csv(os.path.join(HIST_DIR, f"{row.run_id}.csv"))
        ax.plot(hist.turn, hist.time_sigma_ns, label=f"dwell={int(row.dwell_turns)}", alpha=0.7)
    ax.set_xlabel("Turn")
    ax.set_ylabel("RMS bunch length [ns]")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OVERLAY_OUT = os.path.join(DIAG_DIR, "overlay_dwell_turns.png")
    fig.savefig(OVERLAY_OUT, dpi=140)
    print(f"Saved to {OVERLAY_OUT}")

    # ---- 2. min sigma vs dwell_turns (color = turn_of_min) ----
    MINPLOT_OUT = os.path.join(DIAG_DIR, "min_sigma_vs_dwell_turns.png")
    plot_min_sigma_vs_dwell_turns(
        summary_sorted["dwell_turns"].values,
        summary_sorted["min_time_sigma_ns"].values,
        summary_sorted["turn_of_min"].values,
        save_path=MINPLOT_OUT,
    )
    print(f"Saved to {MINPLOT_OUT}")

    # ---- 3. two-panel: min sigma & turn_of_min vs dwell_turns ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.plot(summary_sorted.dwell_turns, summary_sorted.min_time_sigma_ns, "o-", color="tab:blue")
    ax1.set_ylabel("Min RMS bunch length [ns]")
    ax1.set_title(f"phase_jump={summary.phase_jump_deg.iloc[0]:.0f} deg, "
                  f"jump_turns={summary.jump_turns.iloc[0]:.0f}")
    ax1.grid(alpha=0.3)

    ax2.plot(summary_sorted.dwell_turns, summary_sorted.turn_of_min, "o-", color="tab:red")
    ax2.set_xlabel("dwell_turns")
    ax2.set_ylabel("Turn of min")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    PANEL_OUT = os.path.join(DIAG_DIR, "turn_of_min_vs_dwell_turns.png")
    fig.savefig(PANEL_OUT, dpi=140)
    print(f"Saved to {PANEL_OUT}")

    # ---- 4. 2D heatmap: dwell_turns vs turn, colored by sigma_t ----
    histories = {}
    for _, row in summary_sorted.iterrows():
        hist = pd.read_csv(os.path.join(HIST_DIR, f"{row.run_id}.csv"))
        histories[row.dwell_turns] = hist

    dwells = summary_sorted.dwell_turns.values
    # runs have different n_turns since n_turns scales with dwell_turns -- use
    # the shortest run's max turn so every row is fully covered by real data
    max_common_turn = min(histories[d].turn.max() for d in dwells)
    turn_grid = np.linspace(0, max_common_turn, 500)
    Z = np.array([
        np.interp(turn_grid, histories[d].turn, histories[d].time_sigma_ns)
        for d in dwells
    ])

    fig3, ax3 = plt.subplots(figsize=(9, 6))
    n_dwells = len(dwells)
    pcm = ax3.pcolormesh(turn_grid, np.arange(n_dwells), Z, shading="auto",
                          cmap="viridis", norm=LogNorm(vmin=Z.min(), vmax=Z.max()))
    cbar = fig3.colorbar(pcm, ax=ax3, label="RMS bunch length [ns]")
    cbar.set_ticks([2, 3, 5, 8, 10, 20, 30, 40, 60])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    ax3.set_yticks(np.arange(n_dwells))
    ax3.set_yticklabels([f"{int(d)}" for d in dwells])
    ax3.set_xlabel("Turn")
    ax3.set_ylabel("Capture delay (turns)")
    ax3.set_title(f"$\\sigma_t$ vs turn & dwell_turns "
                  f"(phase_jump={summary.phase_jump_deg.iloc[0]:.0f} deg, "
                  f"jump_turns={summary.jump_turns.iloc[0]:.0f})")

    # overlay the capture-jump timing, which shifts with dwell_turns:
    # capture_start_turn = jump_out_start_turn + jump_turns + dwell_turns
    jump_out_start_turn = summary.jump_out_start_turn.iloc[0]
    jump_turns_val = summary.jump_turns.iloc[0]
    capture_turns = jump_out_start_turn + jump_turns_val + dwells
    in_range = capture_turns <= max_common_turn
    ax3.plot(capture_turns[in_range], np.arange(n_dwells)[in_range],
              "k--", linewidth=1.2, alpha=0.7, label="capture jump start")

    # global min
    global_min_idx = summary_sorted["min_time_sigma_ns"].idxmin()
    global_min_row = summary_sorted.loc[global_min_idx]
    print(f"Global min RMS bunch length: {global_min_row.min_time_sigma_ns:.3f} ns "
          f"at dwell={global_min_row.dwell_turns:.0f}, turn={global_min_row.turn_of_min:.0f}")
    min_dwell_idx = np.where(dwells == global_min_row.dwell_turns)[0][0]
    if global_min_row.turn_of_min <= max_common_turn:
        ax3.scatter(global_min_row.turn_of_min, min_dwell_idx, marker="*", s=200,
                    color="red", edgecolor="white", linewidth=0.8, zorder=5,
                    label=f"global min: {global_min_row.min_time_sigma_ns:.2f} ns")

    # runs below 2 ns
    sub_2ns = summary_sorted[summary_sorted["min_time_sigma_ns"] < 2.0].sort_values("min_time_sigma_ns")
    print(f"\n{len(sub_2ns)} runs with min RMS bunch length < 2 ns:")
    for _, row in sub_2ns.iterrows():
        print(f"  dwell={row.dwell_turns:.0f}, turn={row.turn_of_min:.0f}, "
              f"min_sigma={row.min_time_sigma_ns:.3f} ns")
    sub2_valid = sub_2ns[sub_2ns.turn_of_min <= max_common_turn]
    sub2_idx = [np.where(dwells == d)[0][0] for d in sub2_valid.dwell_turns]
    # ax3.scatter(sub2_valid.turn_of_min, sub2_idx, marker="o", s=40,
    #             facecolor="none", edgecolor="white", linewidth=1.2, zorder=4,
    #             label="< 2 ns")

    ax3.legend(loc="upper left", fontsize=8, facecolor="white", framealpha=0.8)
    fig3.tight_layout()

    HEATMAP_OUT = os.path.join(DIAG_DIR, "heatmap_dwell_vs_turn.png")
    fig3.savefig(HEATMAP_OUT, dpi=140)
    print(f"Saved to {HEATMAP_OUT}")