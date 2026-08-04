"""
Adiabatic bunching run.

This file ramps from Vrf_low (20 kV, matching
     non_adiabatic) instead, so the matched-ellipse init + synchrotron
     frequency check have a well-defined bucket to match to at turn 0.
"""

import os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import kinematics
from core.rf import compute_a_coefficient, compute_b_coefficient, check_fixed_point_stability
from core.synchrotron import get_omega_s
from core.separatrix import Separatrix
from core.bunch_init import matched_ellipse_amplitudes, initial_bunch
from core.tracking import track_bunch
from core.diagnostics import save_csv, save_standard_plots
from core.animation import render_animation
from rf_programs.adiabatic import AdiabaticProgram
from core.acceleration import AccelerationProgram
from core.cartoon_plots import render_storyboard
from core.stability import add_stability_columns, report_instabilities 

RNG_SEED = 12345
np.random.seed(RNG_SEED)

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, "results", "adiabatic")
os.makedirs(OUT_DIR, exist_ok=True)

# --- method-specific configuration ---
Vrf_low = 15e3 / 1e9    
Vrf_high = 320e3 / 1e9
ramp_start_turn = 0
ramp_turns = 10000

N = 10000
n_turns = 5000 # max turns or # of turns if stop_after_best_compression = off
eps_l_ns_GeV = 1.35  

# --- acceleration toggle -----------------
ENABLE_ACCELERATION = False
ACCEL_START_TURN = 7000
ACCEL_RAMP_TURNS = 4000
PHI_S_FINAL_DEG = 30

acceleration_program = AccelerationProgram(
    phi_s_final_deg=PHI_S_FINAL_DEG,
    start_turn=ACCEL_START_TURN,
    ramp_turns=ACCEL_RAMP_TURNS,
    enabled=ENABLE_ACCELERATION,
)

voltage_program = AdiabaticProgram(Vrf_low, Vrf_high, ramp_start_turn, ramp_turns)

# --- shared machinery ---
kinematics.print_summary()

a_coef = compute_a_coefficient()
b_coef = compute_b_coefficient(Vrf_low)   
check_fixed_point_stability(a_coef, b_coef)

omega_s, T_s_turns, a_coef, b_coef = get_omega_s(Vrf_low)
print(f"Ramp duration = {ramp_turns} turns = {ramp_turns / T_s_turns:.3f} x T_s "
      f"(>>1 confirms adiabatic)")

if ENABLE_ACCELERATION and ACCEL_START_TURN >= n_turns:
    print(f"NOTE: ACCEL_START_TURN ({ACCEL_START_TURN}) >= n_turns ({n_turns}); "
          f"acceleration is enabled but will never actually start in this run.")

a_t, a_E = matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef)
time0, dE0 = initial_bunch(N, a_t, a_E, method = "uniform", truncate = 6.0)

separatrix = Separatrix(Vrf_max_expected=Vrf_high)

ENABLE_PLOTS = True
ENABLE_ANIMATION = False

# --- run ---
df, snapshots, time_init_for_color = track_bunch(
    time0, dE0, voltage_program, n_turns, a_t, a_E,
    acceleration_program=acceleration_program,
    snapshot_every=100, max_frames=400, stop_after_best_compression=False,
    rows_past_best_to_stop=int(3 * T_s_turns),
)

df = add_stability_columns(df, Nb=1.5e12)
episodes = report_instabilities(df)

print(df['unstable'].sum())   # should be 29 if this matches the earlier CSV
print(episodes)

save_csv(df, f"{OUT_DIR}/diagnostics.csv")

if ENABLE_PLOTS:
    save_standard_plots(df, OUT_DIR)

if ENABLE_ANIMATION:
    render_animation(
        snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
        f"{OUT_DIR}/animation.mp4",
        extra_info=(f"ramp: start={ramp_start_turn}, dur={ramp_turns} turns"
                    + (f" | accel: start={ACCEL_START_TURN}, phi_s->{PHI_S_FINAL_DEG} deg"
                       if ENABLE_ACCELERATION else "")),
    )

print(f"Minimum RMS bunch length: {df.time_sigma_ns.min():.3f} ns "
      f"at turn {df.loc[df.time_sigma_ns.idxmin(), 'turn']:.0f}")
if ENABLE_ACCELERATION:
    print(f"K0: {kinematics.K0:.6f} GeV -> {df.K0_GeV.iloc[-1]:.6f} GeV "
          f"(phi_s_final={PHI_S_FINAL_DEG} deg, start_turn={ACCEL_START_TURN})")
    
N_PANELS = 3
NCOLS    = 3
FIG_WIDTH_IN = 6.5   # set to your report's \textwidth

ENABLE_CARTOON = True

if ENABLE_CARTOON:
    render_storyboard(
        snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
        f"{OUT_DIR}/storyboard.png",
        n_panels=N_PANELS,
        ncols=NCOLS,
        center_on_bunch=False,     # True for a fixed zoomed window
        fig_width_in=FIG_WIDTH_IN,
        # no suptitle, no extra_info: the caption lives in the report
    )

    # vector version for the poster
    render_storyboard(
        snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
        f"{OUT_DIR}/storyboard.pdf",
        n_panels=N_PANELS,
        ncols=NCOLS,
        fig_width_in=FIG_WIDTH_IN,
    )