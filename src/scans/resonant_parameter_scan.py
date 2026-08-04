"""
Single-run resonant method tester.
Set the three knobs below, rerun, read the trace. No scanning, no heatmaps.

Modulation is defined to swing between V_MAX_HARDWARE_KV (top) and a floor set
by MOD_DEPTH toward V_START_KV (bottom):
    MOD_DEPTH = 0   -> no modulation, sits at V_max
    MOD_DEPTH = 1   -> full swing down to V_start
This is converted to the (Vrf_mean, depth) pair ResonantProgram actually
consumes, so the peak voltage can never exceed hardware max regardless of
what you set V_START_KV / MOD_DEPTH to.
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
from core.bunch_init import matched_ellipse_amplitudes, initial_bunch
from core.tracking import track_bunch
from rf_programs.resonant import ResonantProgram
from core.acceleration import AccelerationProgram
import glob
import time


# ================= USER-ADJUSTABLE KNOBS =================
V_START_KV  = 30   # floor voltage the modulation can reach at MOD_DEPTH=1
MOD_DEPTH   = 1.103

RAMP_TURNS  = 3000      # modulation ramp-up turns
# ===========================================================

# fixed sim config
V_max_kv = 320.0
resonance_ratio = 2.0
detuning = 0.0
modulation_start_turn = 0
modulation_phase = 0.0

MOD_DEPTH_MAX_PHYSICAL = V_max_kv / (V_max_kv - V_START_KV)
print(' ')
print("Max Physical Modulation Depth")
print(MOD_DEPTH_MAX_PHYSICAL)
print(' ')
assert 0 <= MOD_DEPTH <= MOD_DEPTH_MAX_PHYSICAL, (
    f"MOD_DEPTH={MOD_DEPTH} exceeds physical limit "
    f"{MOD_DEPTH_MAX_PHYSICAL:.4f} (V_low would go negative)"
)

initial_time_mismatch = 1.05
initial_energy_mismatch = 1.0

N = 10000
n_turns = 8000
eps_l_ns_GeV = 0.95  # From Brendan
emittance_growth_ceiling = 0.14

RNG_SEED = 12345

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, "results", "resonant_single_run")
os.makedirs(OUT_DIR, exist_ok=True)


ROOT_DIR = os.path.join(SRC_DIR, "results", "resonant_single_run")
HIST_DIR = os.path.join(ROOT_DIR, "histories")
os.makedirs(HIST_DIR, exist_ok=True)
SUMMARY_PATH = os.path.join(ROOT_DIR, "scan_summary.csv")

run_id = f"run_{int(time.time()*1000)}"
# --- translate (V_start, depth, ramp) knobs into (Vrf_mean, depth) ---
V_start = V_START_KV * 1e3 / 1e9      # GV
V_max = V_max_kv * 1e3 / 1e9  # GV

V_low = V_max - MOD_DEPTH * (V_max - V_start)
Vrf_mean = (V_max + V_low) / 2
depth = (V_max - V_low) / (2 * Vrf_mean)

mod_stop_turn = n_turns
mod_rampdown_turns = RAMP_TURNS / 4

a_coef = compute_a_coefficient()
b_coef = compute_b_coefficient(Vrf_mean)
check_fixed_point_stability(a_coef, b_coef)
omega_s, T_s_turns, a_coef, b_coef = get_omega_s(Vrf_mean)

b_coef_init = compute_b_coefficient(V_start)   # bunch actually starts at V_start
a_t, a_E = matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef_init)

np.random.seed(RNG_SEED)
time0, dE0 = initial_bunch(N, initial_time_mismatch * a_t, initial_energy_mismatch * a_E)

acceleration_program = AccelerationProgram(
    phi_s_final_deg=30, start_turn=0, ramp_turns=0, enabled=False,
)

voltage_program = ResonantProgram(
    Vrf_mean, depth, omega_s,
    resonance_ratio=resonance_ratio, detuning=detuning,
    start_turn=modulation_start_turn, ramp_turns=RAMP_TURNS,
    mod_phase=modulation_phase,
    stop_turn=mod_stop_turn, rampdown_turns=mod_rampdown_turns,
    V_start_level=V_start,
)

df, _, _ = track_bunch(
    time0.copy(), dE0.copy(), voltage_program, n_turns, a_t, a_E,
    acceleration_program=acceleration_program,
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
    "V_start_kV": V_START_KV,
    "mod_depth": MOD_DEPTH,
    "ramp_turns": RAMP_TURNS,
    "Vrf_mean_kV": Vrf_mean * 1e9 / 1e3,
    "internal_depth": depth,
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

print(f"V_start        = {V_START_KV:.1f} kV")
print(f"MOD_DEPTH      = {MOD_DEPTH:.2f}")
print(f"RAMP_TURNS     = {RAMP_TURNS}")
print(f"-> Vrf_mean    = {Vrf_mean*1e9/1e3:.1f} kV")
print(f"-> depth       = {depth:.3f}")
print(f"-> V range     = [{V_low*1e9/1e3:.1f}, {V_max*1e9/1e3:.1f}] kV")
print(f"min sigma_t    = {min_sigma:.4f} ns @ turn {turn_min:.0f}")
print(f"eps growth     = {eps_growth:.4f}  (valid={valid})")

results_df = pd.DataFrame([{
    "V_start_kV": V_START_KV,
    "mod_depth": MOD_DEPTH,
    "ramp_turns": RAMP_TURNS,
    "Vrf_mean_kV": Vrf_mean * 1e9 / 1e3,
    "internal_depth": depth,
    "min_time_sigma_ns": min_sigma,
    "turn_of_min": turn_min,
    "eps_growth": eps_growth,
    "valid": valid,
}])
results_df.to_csv(f"{OUT_DIR}/single_run_result.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(df.turn, df.time_sigma_ns)
ax.axvline(turn_min, color="red", linestyle="--", alpha=0.6,
           label=f"min @ turn {turn_min:.0f}")
ax.set_xlabel("Turn")
ax.set_ylabel("RMS bunch length [ns]")
ax.set_title(f"V_start={V_START_KV:.0f}kV, depth={MOD_DEPTH:.2f}, ramp={RAMP_TURNS}")
ax.legend()
ax.annotate(f"{min_sigma:.3f} ns",
            xy=(turn_min, min_sigma),
            xytext=(10, 10), textcoords="offset points",
            fontsize=9, color="red")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/bunch_length_trace.png", dpi=140)
plt.close(fig)

print(f"\nSaved trace to {OUT_DIR}/bunch_length_trace.png")