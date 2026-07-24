import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter, NullFormatter


# Define output 
SCANS_DIR = os.path.dirname(os.path.abspath(__file__))         
SRC_DIR   = os.path.dirname(SCANS_DIR)                            
ROOT_DIR  = os.path.join(SRC_DIR, "results", "resonant_single_run")
HIST_DIR  = os.path.join(ROOT_DIR, "histories")
SUMMARY_PATH = os.path.join(ROOT_DIR, "scan_summary.csv")
summary = pd.read_csv(SUMMARY_PATH)
summary = summary.drop_duplicates(
    subset=["mod_depth", "ramp_turns", "V_start_kV"], keep="last"
)

# Mod depth and overlay plot
fig, ax = plt.subplots(figsize=(8, 5))
for _, row in summary.sort_values("mod_depth").iterrows():
    hist = pd.read_csv(os.path.join(HIST_DIR, f"{row.run_id}.csv"))
    ax.plot(hist.turn, hist.time_sigma_ns, label=f"depth={row.mod_depth:.2f}", alpha=0.7)
ax.set_xlabel("Turn")
ax.set_ylabel("RMS bunch length [ns]")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
PLOT_OUT = os.path.join(ROOT_DIR, "overlay_mod_depth.png")
fig.savefig(PLOT_OUT, dpi=140)
print(f"Saved to {PLOT_OUT}")


# Summary plots: min rms bunch length vs modular depth
summary_sorted = summary.sort_values("mod_depth")
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(summary_sorted.mod_depth, summary_sorted.min_time_sigma_ns, "o-", color="tab:blue")
ax2.set_xlabel("Modulation depth")
ax2.set_ylabel("Min RMS bunch length [ns]")
ax2.set_title(f"Min $\\sigma_t$ vs mod depth (V_start={summary.V_start_kV.iloc[0]:.0f} kV, ramp={summary.ramp_turns.iloc[0]:.0f})")
ax2.grid(alpha=0.3)
fig2.tight_layout()
MINPLOT_OUT = os.path.join(ROOT_DIR, "min_sigma_vs_mod_depth.png")
fig2.savefig(MINPLOT_OUT, dpi=140)
print(f"Saved to {MINPLOT_OUT}")
 
SCANS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCANS_DIR)
ROOT_DIR = os.path.join(SRC_DIR, "results", "resonant_single_run")
SUMMARY_PATH = os.path.join(ROOT_DIR, "scan_summary.csv")
 
summary = pd.read_csv(SUMMARY_PATH)
summary = summary.drop_duplicates(
    subset=["mod_depth", "ramp_turns", "V_start_kV"], keep="last"
)
summary_sorted = summary.sort_values("mod_depth")
 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
 
ax1.plot(summary_sorted.mod_depth, summary_sorted.min_time_sigma_ns, "o-", color="tab:blue")
ax1.set_ylabel("Min RMS bunch length [ns]")
ax1.set_title(f"V_start={summary.V_start_kV.iloc[0]:.0f} kV, ramp={summary.ramp_turns.iloc[0]:.0f}")
ax1.grid(alpha=0.3)
 
ax2.plot(summary_sorted.mod_depth, summary_sorted.turn_of_min, "o-", color="tab:red")
ax2.set_xlabel("Modulation depth")
ax2.set_ylabel("Turn of min")
ax2.grid(alpha=0.3)
 
fig.tight_layout()
PLOT_OUT = os.path.join(ROOT_DIR, "turn_of_min_vs_mod_depth.png")
fig.savefig(PLOT_OUT, dpi=140)
print(f"Saved to {PLOT_OUT}")

# 2D heatmap: mod_depth vs turn, colored by RMS bunch length
histories = {}
for _, row in summary_sorted.iterrows():
    hist = pd.read_csv(os.path.join(HIST_DIR, f"{row.run_id}.csv"))
    histories[row.mod_depth] = hist

turn_grid = histories[summary_sorted.mod_depth.iloc[0]].turn.values
depths = summary_sorted.mod_depth.values
Z = np.array([
    np.interp(turn_grid, histories[d].turn, histories[d].time_sigma_ns)
    for d in depths
])

fig3, ax3 = plt.subplots(figsize=(9, 6))
n_depths = len(depths)
pcm = ax3.pcolormesh(turn_grid, np.arange(n_depths), Z, shading="auto",
                      cmap="viridis", norm=LogNorm(vmin=Z.min(), vmax=Z.max()))
cbar = fig3.colorbar(pcm, ax=ax3, label="RMS bunch length [ns]")
cbar.set_ticks([2, 3, 5, 8, 10, 20, 30, 40, 60])
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
cbar.ax.yaxis.set_minor_formatter(NullFormatter())
ax3.set_yticks(np.arange(n_depths))
ax3.set_yticklabels([f"{d:.2f}" for d in depths])
ax3.set_xlabel("Turn")
ax3.set_ylabel("Modulation depth")
ax3.set_title(f"$\\sigma_t$ vs turn & mod depth (V_start={summary.V_start_kV.iloc[0]:.0f} kV, ramp={summary.ramp_turns.iloc[0]:.0f})")

# Global min
global_min_idx = summary_sorted["min_time_sigma_ns"].idxmin()
global_min_row = summary_sorted.loc[global_min_idx]
print(f"Global min RMS bunch length: {global_min_row.min_time_sigma_ns:.3f} ns "
      f"at depth={global_min_row.mod_depth:.3f}, turn={global_min_row.turn_of_min:.0f}")
min_depth_idx = np.where(depths == global_min_row.mod_depth)[0][0]
ax3.scatter(global_min_row.turn_of_min, min_depth_idx, marker="*", s=200,
            color="red", edgecolor="white", linewidth=0.8, zorder=5,
            label=f"global min: {global_min_row.min_time_sigma_ns:.2f} ns")

# Runs below 2 ns
sub_2ns = summary_sorted[summary_sorted["min_time_sigma_ns"] < 2.0].sort_values("min_time_sigma_ns")
print(f"\n{len(sub_2ns)} runs with min RMS bunch length < 2 ns:")
for _, row in sub_2ns.iterrows():
    print(f"  depth={row.mod_depth:.3f}, turn={row.turn_of_min:.0f}, "
          f"min_sigma={row.min_time_sigma_ns:.3f} ns")
sub2_idx = [np.where(depths == d)[0][0] for d in sub_2ns.mod_depth]
ax3.scatter(sub_2ns.turn_of_min, sub2_idx, marker="o", s=40,
            facecolor="none", edgecolor="white", linewidth=1.2, zorder=4,
            label="< 2 ns")

ax3.legend(loc="upper left", fontsize=8, facecolor="white", framealpha=0.8)
fig3.tight_layout()

HEATMAP_OUT = os.path.join(ROOT_DIR, "heatmap_depth_vs_turn.png")
fig3.savefig(HEATMAP_OUT, dpi=140)
print(f"Saved to {HEATMAP_OUT}")
