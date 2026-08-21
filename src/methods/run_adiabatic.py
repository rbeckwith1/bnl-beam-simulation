"""
Adiabatic bunching run.

Ramps from Vrf_low (matching non_adiabatic) rather than 0 kV, so the
matched-ellipse init + synchrotron frequency check have a well-defined
bucket to match to at turn 0.

Ramp shape is LINEAR (constant slew rate), identical to non_adiabatic.
Ramp duration is the only variable separating the two methods.
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
Vrf_low = 5e3 / 1e9
Vrf_high = 320e3 / 1e9
ramp_start_turn = 0     
ramp_turns = 10000

N = 10000
n_turns = 14000  # max turns or # of turns if stop_after_best_compression = off
eps_l_ns_GeV = 1.35

# f_rf / h. Wire this to kinematics if it exposes f_rev directly.
F_REV_HZ = 2.226858e6 / 6

# Set this when the RF engineer confirms the AGS dV/dt limit; None = unknown.
SLEW_LIMIT_KV_PER_MS = None

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

# Adiabaticity: alpha = |d(mu_s)/dn| / mu_s^2, with mu_s = 2*pi/T_s the phase
# advance per turn. The turn-integral of alpha depends only on the endpoints,
# not the ramp shape, so <alpha> below is shape-independent and is the honest
# figure of merit. <alpha> << 1 is adiabatic; <alpha> ~ 1 is not.
alpha_mean = (1.0 - np.sqrt(Vrf_low / Vrf_high)) * T_s_turns / (2 * np.pi * ramp_turns)
slew_kV_per_ms = voltage_program.slew * 1e3 * F_REV_HZ

print(f"Ramp duration = {ramp_turns} turns = {ramp_turns / T_s_turns:.3f} x T_s(Vrf_low) "
      f"= {ramp_turns / F_REV_HZ * 1e3:.2f} ms")
print(f"Mean adiabaticity <alpha> = {alpha_mean:.3f} "
      f"({'adiabatic' if alpha_mean < 0.1 else 'NOT adiabatic -- bunch will be mismatched'})")
print(f"Turns needed for <alpha> = 0.1: "
      f"{(1.0 - np.sqrt(Vrf_low / Vrf_high)) * T_s_turns / (2 * np.pi * 0.1):,.0f}")
print(f"Slew rate = {slew_kV_per_ms:.1f} kV/ms (constant for a linear ramp)")
if SLEW_LIMIT_KV_PER_MS is not None and slew_kV_per_ms > SLEW_LIMIT_KV_PER_MS:
    print(f"  WARNING: exceeds machine limit of {SLEW_LIMIT_KV_PER_MS} kV/ms by "
          f"{slew_kV_per_ms / SLEW_LIMIT_KV_PER_MS:.2f}x -- increase ramp_turns to "
          f"{ramp_turns * slew_kV_per_ms / SLEW_LIMIT_KV_PER_MS:,.0f}")

if ramp_start_turn + ramp_turns > n_turns:
    print(f"WARNING: ramp ends at turn {ramp_start_turn + ramp_turns} but n_turns = {n_turns}; "
          f"the ramp is truncated at V = {voltage_program(n_turns) * 1e6:.1f} kV and never "
          f"reaches Vrf_high.")

if ENABLE_ACCELERATION and ACCEL_START_TURN >= n_turns:
    print(f"NOTE: ACCEL_START_TURN ({ACCEL_START_TURN}) >= n_turns ({n_turns}); "
          f"acceleration is enabled but will never actually start in this run.")

a_t, a_E = matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef)
time0, dE0 = initial_bunch(N, a_t, a_E, method="uniform", truncate=6.0)

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

print(df['unstable'].sum())
print(episodes)

save_csv(df, f"{OUT_DIR}/diagnostics.csv")

if ENABLE_PLOTS:
    save_standard_plots(df, OUT_DIR)

if ENABLE_ANIMATION:
    render_animation(
        snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
        f"{OUT_DIR}/animation.mp4",
        extra_info=(f"linear ramp: start={ramp_start_turn}, dur={ramp_turns} turns, "
                    f"{slew_kV_per_ms:.1f} kV/ms"
                    + (f" | accel: start={ACCEL_START_TURN}, phi_s->{PHI_S_FINAL_DEG} deg"
                       if ENABLE_ACCELERATION else "")),
    )

sigma_matched_ns = df.time_sigma_ns.iloc[0] * (Vrf_low / Vrf_high) ** 0.25
print(f"Minimum RMS bunch length: {df.time_sigma_ns.min():.3f} ns "
      f"at turn {df.loc[df.time_sigma_ns.idxmin(), 'turn']:.0f}")
print(f"Matched sigma_t at Vrf_high = {sigma_matched_ns:.3f} ns "
      f"(sigma_min below this is quadrupole mismatch, not compression)")
if ENABLE_ACCELERATION:
    print(f"K0: {kinematics.K0:.6f} GeV -> {df.K0_GeV.iloc[-1]:.6f} GeV "
          f"(phi_s_final={PHI_S_FINAL_DEG} deg, start_turn={ACCEL_START_TURN})")

N_PANELS = 3
NCOLS = 3
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