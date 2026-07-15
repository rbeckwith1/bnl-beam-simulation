"""
Non-adiabatic bunching run. This is your original non-adiabatic script,
migrated to pull all shared physics from core/ and its voltage program from
rf_programs/. Compare this length to the original ~450-line script -- almost
everything left here is genuinely non-adiabatic-specific configuration.
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
from rf_programs.non_adiabatic import NonAdiabaticProgram

RNG_SEED = 12345
np.random.seed(RNG_SEED)

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, "results", "non_adiabatic")
os.makedirs(OUT_DIR, exist_ok=True)

# --- method-specific configuration (this is the only place these live) ---
Vrf_low = 20e3 / 1e9
Vrf_high = 320e3 / 1e9
jump_start_turn = 2000
jump_turns = 50

N = 10000
n_turns = 10000
eps_l_ns_GeV = 0.95

voltage_program = NonAdiabaticProgram(Vrf_low, Vrf_high, jump_start_turn, jump_turns)

# --- shared machinery ---
kinematics.print_summary()

a_coef = compute_a_coefficient()
b_coef = compute_b_coefficient(Vrf_low)   # linearize about pre-jump voltage
check_fixed_point_stability(a_coef, b_coef)

omega_s, T_s_turns, a_coef, b_coef = get_omega_s(Vrf_low)
print(f"Jump duration = {jump_turns} turns = {jump_turns / T_s_turns:.3f} x T_s "
      f"(<<1 confirms non-adiabatic)")

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
    extra_info=f"jump: start={jump_start_turn}, dur={jump_turns} turns",
)

print(f"Minimum RMS bunch length: {df.time_sigma_ns.min():.3f} ns "
      f"at turn {df.loc[df.time_sigma_ns.idxmin(), 'turn']:.0f}")
