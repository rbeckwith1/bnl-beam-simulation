"""
Single-run UFP release-and-capture method tester.
Set the three knobs below, rerun, read the trace. No scanning, no heatmaps
in this file -- that's what ufp_scan_diagnostics.py (reading scan_summary.csv)
is for.

Mirrors resonant_single_run.py: each run is appended as one row to
scan_summary.csv, plus a full turn-by-turn history CSV, so you can rerun this
script with different knob values and build up a scan the same way you did
for the resonant method.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import kinematics
from core.rf import compute_a_coefficient, compute_b_coefficient, check_fixed_point_stability
from core.synchrotron import get_omega_s
from core.separatrix import Separatrix
from core.bunch_init import matched_ellipse_amplitudes, initial_bunch
from core.tracking import track_bunch
from rf_programs.ufp_capture import UFPReleaseCaptureProgram
import time


# ================= USER-ADJUSTABLE KNOBS =================
DWELL_TURNS     = 600     # turns spent dwelling near the old UFP before capture
PHASE_JUMP_DEG  = 180.0   # size of the RF phase jump (release, then capture)
JUMP_TURNS      = 5       # turns over which each phase jump is ramped
# ===========================================================

# fixed sim config
V_hold_kV = 320.0
phi_s_hold_deg = 0.0
jump_out_start_turn = 200   # let the matched bunch sit quietly first, then release

N = 10000
eps_l_ns_GeV = 0.95  # From Brendan
emittance_growth_ceiling = 0.14
margin_turns = 2000  # extra turns after capture to let things settle

RNG_SEED = 12345

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.join(SRC_DIR, "results", "ufp_scan")
HIST_DIR = os.path.join(ROOT_DIR, "histories")
os.makedirs(HIST_DIR, exist_ok=True)
SUMMARY_PATH = os.path.join(ROOT_DIR, "scan_summary.csv")

run_id = f"run_{int(time.time()*1000)}"

# --- translate knobs into physical quantities ---
V_hold = V_hold_kV * 1e3 / 1e9  # GV
phi_ref_hold = np.pi - np.deg2rad(phi_s_hold_deg)

n_turns = jump_out_start_turn + JUMP_TURNS + DWELL_TURNS + JUMP_TURNS + margin_turns

a_coef = compute_a_coefficient()
b_coef = compute_b_coefficient(V_hold)   # valid: computed at phi_ref=pi, before any jump
check_fixed_point_stability(a_coef, b_coef)

omega_s, T_s_turns, a_coef, b_coef = get_omega_s(V_hold)
a_t, a_E = matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef)

separatrix = Separatrix(Vrf_max_expected=V_hold)
t_ufp_ns = separatrix.unstable_fixed_point_t_ns(phi_ref_hold)
t_ufp_after_jump = separatrix.unstable_fixed_point_t_ns(
    phi_ref_hold + np.deg2rad(PHASE_JUMP_DEG))

np.random.seed(RNG_SEED)
time0, dE0 = initial_bunch(N, a_t, a_E)   # centered at (0, 0) = the SFP

phase_program = UFPReleaseCaptureProgram(
    phi_s_hold_deg=phi_s_hold_deg,
    phase_jump_deg=PHASE_JUMP_DEG,
    jump_out_start_turn=jump_out_start_turn,
    jump_turns=JUMP_TURNS,
    dwell_turns=DWELL_TURNS,
)

def voltage_program(turn):
    return V_hold

df, _, _ = track_bunch(
    time0.copy(), dE0.copy(), voltage_program, n_turns, a_t, a_E,
    acceleration_program=phase_program,
    snapshot_every=None, max_frames=0,
    stop_after_best_compression=False,
)

idx_min = df.time_sigma_ns.idxmin()
min_sigma = df.time_sigma_ns.loc[idx_min]
turn_min = df.turn.loc[idx_min]

eps_init = df.Q1.iloc[0] if "Q1" in df.columns else np.nan
eps_at_min = df.Q1.loc[idx_min] if "Q1" in df.columns else np.nan
eps_growth = (eps_at_min - eps_init) / eps_init if eps_init else np.nan
valid = (eps_growth <= emittance_growth_ceiling) if not np.isnan(eps_growth) else True

# full history for this run
hist_cols = ["turn", "time_sigma_ns"]
if "Q1" in df.columns:
    hist_cols.append("Q1")
df[hist_cols].to_csv(os.path.join(HIST_DIR, f"{run_id}.csv"), index=False)

# summary row, appended
summary_row = pd.DataFrame([{
    "run_id": run_id,
    "dwell_turns": DWELL_TURNS,
    "phase_jump_deg": PHASE_JUMP_DEG,
    "jump_turns": JUMP_TURNS,
    "jump_out_start_turn": jump_out_start_turn,
    "t_ufp_pre_jump_ns": t_ufp_ns,
    "t_ufp_post_jump_ns": t_ufp_after_jump,
    "min_time_sigma_ns": min_sigma,
    "turn_of_min": turn_min,
    "eps_growth": eps_growth,
    "valid": valid,
}])
summary_row.to_csv(
    SUMMARY_PATH,
    mode="a",
    header=not os.path.exists(SUMMARY_PATH),
    index=False,
)

print(f"DWELL_TURNS       = {DWELL_TURNS}")
print(f"PHASE_JUMP_DEG    = {PHASE_JUMP_DEG:.1f}")
print(f"JUMP_TURNS        = {JUMP_TURNS}")
print(f"-> old UFP (pre-jump)  t = {t_ufp_ns:.4f} ns")
print(f"-> new UFP (post-jump) t = {t_ufp_after_jump:.4f} ns (should be ~0)")
print(f"min sigma_t       = {min_sigma:.4f} ns @ turn {turn_min:.0f}")
print(f"eps growth        = {eps_growth:.4f}  (valid={valid})")

results_df = pd.DataFrame([{
    "dwell_turns": DWELL_TURNS,
    "phase_jump_deg": PHASE_JUMP_DEG,
    "jump_turns": JUMP_TURNS,
    "jump_out_start_turn": jump_out_start_turn,
    "min_time_sigma_ns": min_sigma,
    "turn_of_min": turn_min,
    "eps_growth": eps_growth,
    "valid": valid,
}])
results_df.to_csv(f"{ROOT_DIR}/single_run_result.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(df.turn, df.time_sigma_ns)
ax.axvline(turn_min, color="red", linestyle="--", alpha=0.6,
           label=f"min @ turn {turn_min:.0f}")
release_end_turn = jump_out_start_turn + JUMP_TURNS
capture_start_turn = release_end_turn + DWELL_TURNS
ax.axvspan(jump_out_start_turn, release_end_turn, color="tab:orange", alpha=0.15,
           label="release jump")
ax.axvspan(capture_start_turn, capture_start_turn + JUMP_TURNS, color="tab:green",
           alpha=0.15, label="capture jump")
ax.set_xlabel("Turn")
ax.set_ylabel("RMS bunch length [ns]")
ax.set_title(f"dwell={DWELL_TURNS}, phase_jump={PHASE_JUMP_DEG:.0f}deg, "
             f"jump_turns={JUMP_TURNS}")
ax.legend(fontsize=8)
ax.annotate(f"{min_sigma:.3f} ns",
            xy=(turn_min, min_sigma),
            xytext=(10, 10), textcoords="offset points",
            fontsize=9, color="red")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{ROOT_DIR}/bunch_length_trace.png", dpi=140)
plt.close(fig)

print(f"\nSaved trace to {ROOT_DIR}/bunch_length_trace.png")