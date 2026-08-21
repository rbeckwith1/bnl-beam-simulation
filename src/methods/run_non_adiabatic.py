"""
Non-adiabatic bunching run.

Ramp shape is LINEAR (constant slew rate), identical to adiabatic.
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
from rf_programs.non_adiabatic import NonAdiabaticProgram
from core.acceleration import AccelerationProgram
from core.cartoon_plots import render_storyboard
from core.stability import add_stability_columns, report_instabilities

RNG_SEED = 12345
np.random.seed(RNG_SEED)
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, "results", "non_adiabatic")
os.makedirs(OUT_DIR, exist_ok=True)

# --- method-specific configuration (this is the only place these live) ---
Vrf_low = 1e3 / 1e9
Vrf_high = 320e3 / 1e9
jump_start_turn = 0          # MUST match adiabatic's ramp_start_turn
jump_turns = 5            # machine limit? -> check slew warning below
N = 10000
n_turns = 690
eps_l_ns_GeV = 1.35

# f_rf / h. Wire this to kinematics if it exposes f_rev directly.
F_REV_HZ = 2.226858e6 / 6

# Set this when the RF engineer confirms the AGS dV/dt limit; None = unknown.
SLEW_LIMIT_KV_PER_MS = None

# --- acceleration toggle -------------------------------------------------
ENABLE_ACCELERATION = False
ACCEL_START_TURN = jump_start_turn + jump_turns + 2000
ACCEL_RAMP_TURNS = 2000
PHI_S_FINAL_DEG = 30
acceleration_program = AccelerationProgram(
    phi_s_final_deg=PHI_S_FINAL_DEG,
    start_turn=ACCEL_START_TURN,
    ramp_turns=ACCEL_RAMP_TURNS,
    enabled=ENABLE_ACCELERATION,
)
voltage_program = NonAdiabaticProgram(Vrf_low, Vrf_high, jump_start_turn, jump_turns)

# --- shared machinery ---
kinematics.print_summary()
a_coef = compute_a_coefficient()
b_coef = compute_b_coefficient(Vrf_low)
check_fixed_point_stability(a_coef, b_coef)
omega_s, T_s_turns, a_coef, b_coef = get_omega_s(Vrf_low)

# Adiabaticity: alpha = |d(mu_s)/dn| / mu_s^2, with mu_s = 2*pi/T_s the phase
# advance per turn. The turn-integral of alpha depends only on the endpoints,
# not the ramp shape, so <alpha> below is shape-independent and is the honest
# figure of merit. <alpha> >> 1 is the non-adiabatic regime.
alpha_mean = (1.0 - np.sqrt(Vrf_low / Vrf_high)) * T_s_turns / (2 * np.pi * jump_turns)
slew_kV_per_ms = voltage_program.slew * 1e3 * F_REV_HZ

print(f"Jump duration = {jump_turns} turns = {jump_turns / T_s_turns:.3f} x T_s(Vrf_low) "
      f"= {jump_turns / F_REV_HZ * 1e3:.2f} ms")
print(f"Mean adiabaticity <alpha> = {alpha_mean:.3f} "
      f"({'NOT non-adiabatic -- alpha is small' if alpha_mean < 0.1 else 'non-adiabatic'})")
print(f"Slew rate = {slew_kV_per_ms:.1f} kV/ms (constant for a linear ramp)")
if SLEW_LIMIT_KV_PER_MS is not None and slew_kV_per_ms > SLEW_LIMIT_KV_PER_MS:
    print(f"  WARNING: exceeds machine limit of {SLEW_LIMIT_KV_PER_MS} kV/ms by "
          f"{slew_kV_per_ms / SLEW_LIMIT_KV_PER_MS:.2f}x -- the shortest feasible jump is "
          f"{jump_turns * slew_kV_per_ms / SLEW_LIMIT_KV_PER_MS:,.0f} turns")

if jump_start_turn + jump_turns > n_turns:
    print(f"WARNING: jump ends at turn {jump_start_turn + jump_turns} but n_turns = {n_turns}; "
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
    snapshot_every=10, max_frames=400,
    stop_after_best_compression=False,
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
        extra_info=(f"linear jump: start={jump_start_turn}, dur={jump_turns} turns, "
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


N_PANELS = 3  # how many turns to show
ENABLE_CARTOON = True

if ENABLE_CARTOON:
    render_storyboard(
        snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
        f"{OUT_DIR}/storyboard.png",
        n_panels=N_PANELS,
        ncols=N_PANELS,                   # ncols == n_panels forces a single inline row
        center_on_bunch=False,            # set True if you want a fixed zoomed window instead
        suptitle="Non-adiabatic bunching",
        extra_info=(f"linear jump: start={jump_start_turn}, dur={jump_turns} turns, "
                    f"{slew_kV_per_ms:.1f} kV/ms"
                    + (f" | accel: start={ACCEL_START_TURN}, phi_s->{PHI_S_FINAL_DEG} deg"
                       if ENABLE_ACCELERATION else "")),
    )
    # vector version for print quality on the poster:
    render_storyboard(
        snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
        f"{OUT_DIR}/storyboard.pdf",
        n_panels=N_PANELS, ncols=N_PANELS,
        suptitle="Non-adiabatic bunching",
    )