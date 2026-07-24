# resonant_scan_diagnostics.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_min_sigma_vs_mod_depth(mod_depths, min_sigmas, turns_of_min,
                                  highlight_range=None, save_path=None):
    """
    Pure plotting function — knows nothing about files or paths.
    mod_depths, min_sigmas, turns_of_min : 1D arrays, same length, sorted by mod_depth
    highlight_range : optional (lo, hi) tuple to shade the validated smooth segment
    """
    mod_depths = np.asarray(mod_depths)
    min_sigmas = np.asarray(min_sigmas)
    turns_of_min = np.asarray(turns_of_min)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if highlight_range is not None:
        ax.axvspan(*highlight_range, color='gray', alpha=0.12, zorder=0)

    points = np.array([mod_depths, min_sigmas]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', zorder=1)
    lc.set_array(turns_of_min[:-1])
    ax.add_collection(lc)

    sc = ax.scatter(mod_depths, min_sigmas, c=turns_of_min, cmap='viridis',
                     s=45, edgecolor='k', linewidth=0.4, zorder=2)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('turn_of_min')

    ax.set_xlabel('mod_depth')
    ax.set_ylabel('min $\\sigma_t$ [ns]')
    ax.set_title('Resonant compression: min $\\sigma_t$ vs mod_depth\n(color = turn at which minimum occurred)')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig, ax


# ---- script logic: only runs when you execute this file directly ----
if __name__ == "__main__":
    SCANS_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR   = os.path.dirname(SCANS_DIR)

    SINGLE_RUN_DIR = os.path.join(SRC_DIR, "results", "resonant_single_run")
    SUMMARY_PATH   = os.path.join(SINGLE_RUN_DIR, "scan_summary.csv")

    DIAG_DIR = os.path.join(SRC_DIR, "results", "resonant_scan_diagnostics")
    os.makedirs(DIAG_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(DIAG_DIR, "min_sigma_vs_mod_depth.png")

    summary = pd.read_csv(SUMMARY_PATH)
    summary = summary.drop_duplicates(subset="mod_depth").sort_values("mod_depth")  # adjust to your actual dedup logic

    plot_min_sigma_vs_mod_depth(
        summary["mod_depth"].values,
        summary["min_time_sigma_ns"].values,
        summary["turn_of_min"].values,
        highlight_range=(0.70, 0.90),
        save_path=SAVE_PATH,
    )