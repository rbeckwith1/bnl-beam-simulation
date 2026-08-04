"""
2D scan: modulation_depth x modulation_ramp_turns (resonant method)
Reuses the single-run script's setup, wraps it in a loop, collects
min bunch length + turn-of-min + emittance growth for each (depth, ramp) pair.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import kinematics
from core.rf import compute_a_coefficient, compute_b_coefficient, check_fixed_point_stability
from core.synchrotron import get_omega_s
from core.bunch_init import matched_ellipse_amplitudes, initial_bunch
from core.tracking import track_bunch
from rf_programs.resonant import ResonantProgram
from core.acceleration import AccelerationProgram

RNG_SEED = 12345

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, "results", "resonant_scan")
os.makedirs(OUT_DIR, exist_ok=True)

# --- fixed config 
resonance_ratio = 2.0
detuning = 0.0
modulation_start_turn = 0
modulation_phase = 0.0

initial_time_mismatch = 1.05
initial_energy_mismatch = 1.0

N = 10000
n_turns = 10000
eps_l_ns_GeV = 0.95 # From Brendan

emittance_growth_ceiling = 0.14
ramp_values  = np.array([2000, 2500, 3000, 3500, 4000])

a_coef = compute_a_coefficient()

acceleration_program = AccelerationProgram(
    phi_s_final_deg=30, start_turn=0, ramp_turns=0, enabled=False,
)

results = []
bunch_length_traces = {}

V_MAX_HARDWARE = 320e3 / 1e9  # GV

V_start_values = np.array([50e3, 100e3, 150e3]) / 1e9  # GV

for V_start in V_start_values:
    Vrf_mean = V_start
    depth_max = V_MAX_HARDWARE / V_start - 1.0
    depth_values = np.linspace(0.1, depth_max, 5)

    b_coef = compute_b_coefficient(Vrf_mean)
    check_fixed_point_stability(a_coef, b_coef)
    omega_s, T_s_turns, a_coef, b_coef = get_omega_s(Vrf_mean)
    a_t, a_E = matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef)

    np.random.seed(RNG_SEED)
    time0, dE0 = initial_bunch(N, initial_time_mismatch * a_t, initial_energy_mismatch * a_E)

    for depth in depth_values:
        for ramp in ramp_values:
            
            mod_stop_turn = n_turns
            mod_rampdown_turns = ramp / 4
    
            voltage_program = ResonantProgram(
                Vrf_mean, depth, omega_s,
                resonance_ratio=resonance_ratio, detuning=detuning,
                start_turn=modulation_start_turn, ramp_turns=ramp,
                mod_phase=modulation_phase,
                stop_turn=mod_stop_turn, rampdown_turns=mod_rampdown_turns,
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
    
            bunch_length_traces[(V_start, depth, ramp)] = (df.turn.values, df.time_sigma_ns.values)
    
            eps_init = df.Q1.iloc[0] if "Q1" in df.columns else np.nan
            eps_at_min = df.Q1.loc[idx_min] if "Q1" in df.columns else np.nan
            eps_growth = (eps_at_min - eps_init) / eps_init if eps_init else np.nan
    
            results.append({
            "Vrf_mean_GV": Vrf_mean,          # add this — matches trace dict key exactly
            "depth": depth,
            "ramp_turns": ramp,
            "Vrf_mean_kV": Vrf_mean * 1e9 / 1e3,
            "min_time_sigma_ns": min_sigma,
            "turn_of_min": turn_min,
            "eps_growth": eps_growth,
            "valid": (eps_growth <= emittance_growth_ceiling) if not np.isnan(eps_growth) else True,
        })
        
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/scan_depth_ramp.csv", index=False)

# make sure this column exists — if you named it differently, adjust here
V_START_COL = "Vrf_mean_kV"  # or "Vrf_mean_kV", whatever you actually called it

for v_start_val, group in results_df.groupby(V_START_COL):

    pivot_sigma = group.pivot(index="depth", columns="ramp_turns", values="min_time_sigma_ns")
    pivot_valid = group.pivot(index="depth", columns="ramp_turns", values="valid")

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot_sigma.values, aspect="auto", origin="lower",
                    extent=[group.ramp_turns.min(), group.ramp_turns.max(),
                            group.depth.min(), group.depth.max()])
    fig.colorbar(im, ax=ax, label="min RMS bunch length [ns]")
    ax.set_xlabel("modulation_ramp_turns")
    ax.set_ylabel("modulation_depth")
    ax.set_title(f"Min bunch length vs. (depth, ramp_turns) — V_start={v_start_val:.0f} kV")

    for depth in pivot_sigma.index:
        for ramp in pivot_sigma.columns:
            if not pivot_valid.loc[depth, ramp]:
                ax.text(ramp, depth, "X", ha="center", va="center", color="red", fontsize=12)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/heatmap_min_sigma_Vstart{v_start_val:.0f}.png", dpi=140)
    plt.close(fig)

    pivot_turn = group.pivot(index="depth", columns="ramp_turns", values="turn_of_min")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot_turn.values, aspect="auto", origin="lower",
                    extent=[group.ramp_turns.min(), group.ramp_turns.max(),
                            group.depth.min(), group.depth.max()])
    fig.colorbar(im, ax=ax, label="turn of min bunch length")
    ax.set_xlabel("modulation_ramp_turns")
    ax.set_ylabel("modulation_depth")
    ax.set_title(f"Turn-of-minimum vs. (depth, ramp_turns) — V_start={v_start_val:.0f} kV")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/heatmap_turn_of_min_Vstart{v_start_val:.0f}.png", dpi=140)
    plt.close(fig)

# --- overall best across all V_start slabs ---
best_row = results_df.loc[results_df[results_df.valid].min_time_sigma_ns.idxmin()]
print("\nBest valid combination overall:")
print(best_row)

best_key = (best_row["Vrf_mean_GV"], best_row.depth, best_row.ramp_turns)
best_turn, best_sigma = bunch_length_traces[best_key]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(best_turn, best_sigma)
ax.axvline(best_row.turn_of_min, color="red", linestyle="--", alpha=0.6,
           label=f"min @ turn {best_row.turn_of_min:.0f}")
ax.set_xlabel("Turn")
ax.set_ylabel("RMS bunch length [ns]")
ax.set_title(f"Best: V_start={best_row[V_START_COL]:.0f}kV, depth={best_row.depth:.2f}, ramp={best_row.ramp_turns:.0f}")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/best_bunch_length_trace.png", dpi=140)
plt.close(fig)

print(f"\nSaved best-combination trace to {OUT_DIR}/best_bunch_length_trace.png")