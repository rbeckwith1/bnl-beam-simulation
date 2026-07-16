""""
Resonant (parametric) bunching run. Migrated from the original standalone
script, following the same pattern as run_non_adiabatic.py: shared physics
comes from core/, and the only things that differ per-method are the
voltage program (rf_programs/resonant.py) and the config block below.

Physical picture: a matched bunch's envelope (quadrupole/breathing) degree
of freedom can be parametrically driven when the RF voltage is modulated at
omega_mod close to 2*omega_s (Mathieu/Hill-type resonance). See
rf_programs/resonant.py for the exact voltage-program formula and the NOTE
on modulation_phase there -- growth vs. suppression is not knowable a
priori and must be checked against the Q1/Q2/theta_Q diagnostics below.

Acceleration (the dK0_turn / phi_s energy ramp) is wired in via
core/acceleration.py -- see ENABLE_ACCELERATION below. It's a toggle, not a
separate code path: core/tracking.py always builds a ReferenceParticle and
always asks the acceleration_program for (dK0, phi_s, phi_ref) every turn,
so ENABLE_ACCELERATION=False reproduces pure resonant modulation (K0 fixed)
exactly, and =True layers an energy ramp on top of the modulation. NOTE:
this combination (accelerate WHILE parametrically modulating) hasn't been
studied in this scaffold before -- the Q1/Q2/theta_Q envelope diagnostics
and the separatrix (now phi_s-aware, see core/separatrix.py) still apply,
but whether the resonance condition (omega_mod = 2*omega_s) stays satisfied
as the bucket shrinks under acceleration is an open question, not something
this file verifies for you.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from rf_programs.resonant import ResonantProgram
from core.acceleration import AccelerationProgram

RNG_SEED = 12345
np.random.seed(RNG_SEED)

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, "results", "resonant")
os.makedirs(OUT_DIR, exist_ok=True)

# --- method-specific configuration (this is the only place these live) ---
Vrf_mean = 240e3 / 1e9           # mean RF voltage used for the modulated run [GeV]
Vrf_max_machine = 320e3 / 1e9     # hardware ceiling, used only to size the separatrix grid generously

modulation_depth = 0.15           # fractional modulation depth (10-20% suggested)
resonance_ratio = 2.0             # omega_mod = resonance_ratio * omega_s * (1+detuning)
detuning = 0.0                    # fractional detuning of the modulation frequency
modulation_start_turn = 2000
modulation_ramp_turns = 12000     # turns over which depth ramps 0 -> modulation_depth
                                   # (0 = instantaneous switch-on; >0 = smooth ramp)
modulation_phase = 0.0            # modulation phase offset [rad] -- see NOTE in rf_programs/resonant.py

# Initial bunch mismatch (optional envelope-oscillation seed)
initial_time_mismatch = 1.05      # >1.0 makes the initial ellipse a monopole+quadrupole seed in t
initial_energy_mismatch = 1.0     # keep at 1.0 to isolate the effect of the time mismatch

N = 10000
n_turns = 30000
eps_l_ns_GeV = 0.95 # from Brendan

# --- acceleration toggle -------------------------------------------------
# Flip ENABLE_ACCELERATION to turn the reference-particle energy ramp on/off.
# ACCEL_START_TURN defaults to after the modulation ramp has fully switched
# on (modulation_start_turn + modulation_ramp_turns), plus slack, so you're
# not trying to grow/damp the envelope resonance and ramp energy at the
# same time on your first pass -- shift it earlier once you actually want
# to study that overlap.
ENABLE_ACCELERATION = False
ACCEL_START_TURN = modulation_start_turn + modulation_ramp_turns + 2000
ACCEL_RAMP_TURNS = 5000
PHI_S_FINAL_DEG = 30

acceleration_program = AccelerationProgram(
    phi_s_final_deg=PHI_S_FINAL_DEG,
    start_turn=ACCEL_START_TURN,
    ramp_turns=ACCEL_RAMP_TURNS,
    enabled=ENABLE_ACCELERATION,
)

# --- shared machinery ---
kinematics.print_summary()

a_coef = compute_a_coefficient()
b_coef = compute_b_coefficient(Vrf_mean)
check_fixed_point_stability(a_coef, b_coef)

omega_s, T_s_turns, a_coef, b_coef = get_omega_s(Vrf_mean)

voltage_program = ResonantProgram(
    Vrf_mean, modulation_depth, omega_s,
    resonance_ratio=resonance_ratio, detuning=detuning,
    start_turn=modulation_start_turn, ramp_turns=modulation_ramp_turns,
    mod_phase=modulation_phase,
)

print(f"Modulation target: omega_mod = {resonance_ratio}*omega_s*(1+{detuning}) "
      f"= {voltage_program.omega_mod:.6e} rad/turn")
print(f"Modulation period: {2.0 * np.pi / voltage_program.omega_mod:.3f} turns "
      f"(compare to T_s/2 = {T_s_turns / 2:.3f} turns)")

if ENABLE_ACCELERATION and ACCEL_START_TURN >= n_turns:
    print(f"NOTE: ACCEL_START_TURN ({ACCEL_START_TURN}) >= n_turns ({n_turns}); "
          f"acceleration is enabled but will never actually start in this run.")

# Matched-ellipse amplitudes: derived from eps_l via the shared
# matched_ellipse_amplitudes() helper (same convention as
# run_non_adiabatic.py), rather than setting a_t directly. This linear
# relation is used for INITIALIZATION only -- the tracking itself is fully
# nonlinear.
a_t, a_E = matched_ellipse_amplitudes(eps_l_ns_GeV, a_coef, b_coef)
print(f"Matched-ellipse initial amplitudes: a_t = {a_t:.3f} ns, "
      f"a_E = {a_E * 1e3:.4f} MeV")
print(f"Initial mismatch factors: time x{initial_time_mismatch}, "
      f"energy x{initial_energy_mismatch}")

time0, dE0 = initial_bunch(N, initial_time_mismatch * a_t, initial_energy_mismatch * a_E)

separatrix = Separatrix(Vrf_max_expected=Vrf_max_machine * (1.0 + modulation_depth))

# --- run ---
df, snapshots, time_init_for_color = track_bunch(
    time0, dE0, voltage_program, n_turns, a_t, a_E,
    acceleration_program=acceleration_program,
    snapshot_every=10, max_frames=400,
    stop_after_best_compression=False,
)

save_csv(df, f"{OUT_DIR}/diagnostics.csv")
save_standard_plots(df, OUT_DIR)
render_animation(
    snapshots, time_init_for_color, a_t, a_E, separatrix, T_s_turns,
    f"{OUT_DIR}/animation.mp4",
    extra_info=(f"mod: start={modulation_start_turn}, "
                f"ramp={modulation_ramp_turns} turns, depth={modulation_depth}"
                + (f" | accel: start={ACCEL_START_TURN}, phi_s->{PHI_S_FINAL_DEG} deg"
                   if ENABLE_ACCELERATION else "")),
)

# --- resonant-specific diagnostic: modulation phase vs. 2*theta_Q -------
# Not part of the shared plot_1..7 set (core/diagnostics.py) since it's only
# meaningful for a *modulated* voltage program; this is the plot that
# actually tells you whether modulation_phase landed on the driving or the
# damping quadrature.
mod_phase_series = np.where(
    df.turn >= modulation_start_turn,
    (voltage_program.omega_mod * (df.turn - modulation_start_turn)
     + modulation_phase) % (2 * np.pi),
    np.nan,
)
two_thetaQ_wrapped = (2.0 * df.theta_Q) % (2 * np.pi)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(df.turn, mod_phase_series, label="RF modulation phase (mod 2pi)")
ax.plot(df.turn, two_thetaQ_wrapped, label="2*theta_Q (mod 2pi)")
ax.set_xlabel("Turn")
ax.set_ylabel("Phase [rad]")
ax.set_title("RF modulation phase vs. 2*(bunch orientation phase)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/plot_8_phase_comparison.png", dpi=140)
plt.close(fig)
print(f"  saved {OUT_DIR}/plot_8_phase_comparison.png")

print(f"Minimum RMS bunch length: {df.time_sigma_ns.min():.3f} ns "
      f"at turn {df.loc[df.time_sigma_ns.idxmin(), 'turn']:.0f}")

if ENABLE_ACCELERATION:
    print(f"K0: {kinematics.K0:.6f} GeV -> {df.K0_GeV.iloc[-1]:.6f} GeV "
          f"(phi_s_final={PHI_S_FINAL_DEG} deg, start_turn={ACCEL_START_TURN})")