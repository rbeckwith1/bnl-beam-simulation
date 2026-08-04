"""
Non-adiabatic bunching optimization scan.

Primary optimization axis: Vrf_low (the voltage the bunch is matched to
BEFORE the jump). Vrf_high is pinned at the hardware ceiling, so the
voltage ratio Vrf_high / Vrf_low is what actually sets how mismatched
(and therefore how compressed) the bunch becomes after the jump.

jump_turns is scanned too, but only as a robustness/sanity axis: once
jump_turns << T_s (synchrotron period at Vrf_low) you're safely in the
non-adiabatic regime and the result should plateau. It is NOT expected
to behave like a free compression knob the way Vrf_low does -- if you
see monotonic improvement with faster jumps that never plateaus, that's
worth a second look (could mean T_s at Vrf_low is being computed wrong,
or n_turns is too short to find the real minimum).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rf import compute_a_coefficient, compute_b_coefficient, check_fixed_point_stability
from core.synchrotron import get_omega_s
from core.bunch_init import matched_ellipse_amplitudes, initial_bunch
from core.tracking import track_bunch
from rf_programs.non_adiabatic import NonAdiabaticProgram
from core.acceleration import AccelerationProgram

# ================= SCAN GRID =================
VRF_HIGH_KV      = 320.0                          # hardware ceiling, fixed
VRF_LOW_KV_LIST  = np.linspace(1, 200, 19)       # PRIMARY axis
JUMP_TURNS_LIST = [5, 20, 80, 320, 1280, 5120, 20480]
JUMP_START_TURN  = 500
THRESHOLD_NS     = 2.0            # highlight region achieving sub-2ns compression
# ===============================================

N = 10000
n_turns = 3000
eps_l_ns_GeV = 1.35
emittance_growth_ceiling = 0.14 # arbitrary
RNG_SEED = 12345 #random number generator - help generate particles with random positions/energies for gaussian

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, "results", "non_adiabatic_scan")
HIST_DIR = os.path.join(OUT_DIR, "histories")
os.makedirs(HIST_DIR, exist_ok=True)
SUMMARY_PATH = os.path.join(OUT_DIR, "scan_summary.csv")

Vrf_high = VRF_HIGH_KV * 1e3 / 1e9  # GV

rows = []
for Vrf_low_kV in VRF_LOW_KV_LIST:
    Vrf_low = Vrf_low_kV * 1e3 / 1e9

    # bunch is matched to Vrf_low BEFORE the jump -- b_coef/a_t/a_E must be
    # computed at Vrf_low, not Vrf_high or any mean (same bug class as the
    # resonant Vrf_mean/V_start mixup -- worth double-checking here too).
    a_coef = compute_a_coefficient()
    b_coef_low = compute_b_coefficient(Vrf_low)
    check_fixed_point_stability(a_coef, b_coef_low)
    omega_s_low, T_s_turns_low, _, _ = get_omega_s(Vrf_low)

    a_t, a_E = matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef_low)

    for jump_turns in JUMP_TURNS_LIST:
        run_id = f"run_{int(time.time()*1e6)}"
        np.random.seed(RNG_SEED)
        time0, dE0 = initial_bunch(N, a_t, a_E)

        acceleration_program = AccelerationProgram(
            phi_s_final_deg=30, start_turn=0, ramp_turns=0, enabled=False,
        )
        voltage_program = NonAdiabaticProgram(
            Vrf_low, Vrf_high, JUMP_START_TURN, jump_turns,
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

        jump_over_Ts = jump_turns / T_s_turns_low
        hit_end_of_window = (n_turns - turn_min) < 5  # min landed at/near the edge

        row = {
            "run_id": run_id,
            "Vrf_low_kV": Vrf_low_kV,
            "Vrf_high_kV": VRF_HIGH_KV,
            "voltage_ratio": VRF_HIGH_KV / Vrf_low_kV,
            "jump_turns": jump_turns,
            "jump_start_turn": JUMP_START_TURN,
            "T_s_turns_low": T_s_turns_low,
            "jump_over_Ts": jump_over_Ts,
            "min_time_sigma_ns": min_sigma,
            "turn_of_min": turn_min,
            "hit_end_of_window": hit_end_of_window,
            "eps_growth": eps_growth,
            "valid": valid,
        }
        rows.append(row)

        hist_cols = ["turn", "time_sigma_ns"]
        if "Q1" in df.columns:
            hist_cols.append("Q1")
        df[hist_cols].to_csv(os.path.join(HIST_DIR, f"{run_id}.csv"), index=False)

        flag = "  <-- hit end of window, increase n_turns" if hit_end_of_window else ""
        print(f"Vrf_low={Vrf_low_kV:6.1f} kV  jump_turns={jump_turns:4d} "
              f"(={jump_over_Ts:.3f} T_s)  -> min sigma_t = {min_sigma:.4f} ns "
              f"@ turn {turn_min:.0f}, eps_growth={eps_growth:.4f}, valid={valid}{flag}")

summary_df = pd.DataFrame(rows)
summary_df["under_threshold"] = summary_df["min_time_sigma_ns"] < THRESHOLD_NS
summary_df.to_csv(SUMMARY_PATH, index=False)
n_hits = summary_df["under_threshold"].sum()
print(f"\nSaved scan summary to {SUMMARY_PATH}")
print(f"{n_hits} / {len(summary_df)} grid points achieve < {THRESHOLD_NS:.1f} ns")

# ================= DIAGNOSTIC PLOTS =================

# 1) PRIMARY: best achievable sigma (best over jump_turns) vs Vrf_low
valid_df = summary_df[summary_df.valid]
best_per_voltage = valid_df.loc[
    valid_df.groupby("Vrf_low_kV")["min_time_sigma_ns"].idxmin()
].sort_values("Vrf_low_kV")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(best_per_voltage.Vrf_low_kV, best_per_voltage.min_time_sigma_ns, "o-")
ax.set_xlabel("Vrf_low [kV]  (starting / matched voltage before the jump)")
ax.set_ylabel("Best achievable RMS bunch length [ns]")
ax.set_title(f"Non-adiabatic: compression vs starting voltage (Vrf_high={VRF_HIGH_KV:.0f} kV)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sigma_vs_Vrf_low.png"), dpi=140)
plt.close(fig)

# 2) SECONDARY: sigma vs jump_turns for a few representative Vrf_low values
#    -- expect these curves to flatten out at small jump_turns
fig, ax = plt.subplots(figsize=(7, 4.5))
sample_voltages = sorted(summary_df.Vrf_low_kV.unique())[::4]
for v in sample_voltages:
    sub = summary_df[summary_df.Vrf_low_kV == v].sort_values("jump_turns")
    ax.plot(sub.jump_turns, sub.min_time_sigma_ns, "o-", label=f"Vrf_low={v:.0f} kV")
ax.set_xlabel("jump_turns")
ax.set_ylabel("Min RMS bunch length [ns]")
ax.set_title("Robustness check: compression vs jump duration")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sigma_vs_jump_turns.png"), dpi=140)
plt.close(fig)

# 3) Full 2D picture
pivot = summary_df.pivot(index="Vrf_low_kV", columns="jump_turns", values="min_time_sigma_ns")
X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)  # X=jump_turns, Y=Vrf_low_kV
Z = pivot.values

fig, ax = plt.subplots(figsize=(7, 5))
mesh = ax.pcolormesh(X, Y, Z, shading="nearest")
fig.colorbar(mesh, ax=ax, label="Min RMS bunch length [ns]")

ax.set_xscale("log")
ax.set_xlabel("jump_turns")
ax.set_ylabel("Vrf_low [kV]")
ax.set_title("Min RMS bunch length [ns]")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "scan_heatmap.png"), dpi=140)
plt.close(fig)