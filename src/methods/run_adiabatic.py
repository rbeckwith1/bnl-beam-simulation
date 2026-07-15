"""
Adiabatic bunching run. Ported from the original rf_bucket_motion.py
(quadratic Vrf ramp, 0 -> 320 kV over 100000 turns), migrated to pull all
shared physics from core/ and its voltage program from rf_programs/, same
pattern as run_non_adiabatic.py.

Two things were changed relative to the original script -- see the inline
NOTE comments below and rf_programs/adiabatic.py for why:
  1. gamma_t / h: the original script hardcoded gamma_t=8.667, h=12, which
     conflict with core/constants.py's gamma_t=8.45, h=6 (the values already
     in use by run_non_adiabatic.py). Machine constants are supposed to be
     shared across all three methods, so this file uses the core/constants.py
     values -- CONFIRM WITH DR. BROOKS which pair is correct for Run 24 and
     fix core/constants.py directly if it's the script's values, not these.
  2. Vrf_initial: the original ramped from 0 kV with a raw uniform-box
     initial distribution. This file ramps from Vrf_low (20 kV, matching
     non_adiabatic) instead, so the matched-ellipse init + synchrotron
     frequency check have a well-defined bucket to match to at turn 0.

Also dropped: the unused acceleration-ramp branch (dK0_turn / phi_s energy
ramp) from the original script -- accel_start_turn was set to 1e11 turns, so
it never fired, and core/tracking.py currently assumes fixed K0 (pure
bunching, no acceleration) like the other two methods.
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

RNG_SEED = 12345
np.random.seed(RNG_SEED)

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, "results", "adiabatic")
os.makedirs(OUT_DIR, exist_ok=True)

# --- method-specific configuration (this is the only place these live) ---
Vrf_low = 20e3 / 1e9     # GeV; NOTE: was 0 kV in the original script -- see module docstring
Vrf_high = 320e3 / 1e9
ramp_start_turn = 0
ramp_turns = 100000

N = 10000
n_turns = 120000   # matches original: turns_per_frame(100) * n_frames(1200)
eps_l_ns_GeV = 0.95 * (2 / 3)   # TODO(Rosalyn): replace with confirmed Run 24 value

voltage_program = AdiabaticProgram(Vrf_low, Vrf_high, ramp_start_turn, ramp_turns)

# --- shared machinery ---
kinematics.print_summary()

a_coef = compute_a_coefficient()
b_coef = compute_b_coefficient(Vrf_low)   # linearize about pre-ramp voltage
check_fixed_point_stability(a_coef, b_coef)

omega_s, T_s_turns, a_coef, b_coef = get_omega_s(Vrf_low)
print(f"Ramp duration = {ramp_turns} turns = {ramp_turns / T_s_turns:.3f} x T_s "
      f"(>>1 confirms adiabatic)")

a_t, a_E = matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef)
time0, dE0 = initial_bunch(N, a_t, a_E)

separatrix = Separatrix(Vrf_max_expected=Vrf_high)

# --- run ---
df, snapshots, time_init_for_color = track_bunch(
    time0, dE0, voltage_program, n_turns, a_t, a_E,
    snapshot_every=10, max_frames=400,
    min_turn_for_stop=jump_start_turn + jump_turns,
)

save_csv(df, f"{OUT_DIR}/diagnostics.csv")
save_standard_plots(df, OUT_DIR)
render_animation(
    snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
    f"{OUT_DIR}/animation.mp4",
    extra_info=f"ramp: start={ramp_start_turn}, dur={ramp_turns} turns",
)

print(f"Minimum RMS bunch length: {df.time_sigma_ns.min():.3f} ns "
      f"at turn {df.loc[df.time_sigma_ns.idxmin(), 'turn']:.0f}")