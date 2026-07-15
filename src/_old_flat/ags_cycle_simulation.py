"""
AGS 6<->12 squeeze/merge -- simplified single-harmonic bucket-reshaping model
==============================================================================

Scope ( simplified, per BNL C-A/AP/685 comparison)
----------------------------------------------------------------
This is NOT a multi-cavity, multi-harmonic reconstruction of the real RF
program (that would require digitized voltage-vs-time values per cavity
group from Fig. 12 of the note, which aren't available here). Instead:

    - ONE representative harmonic (h=12, already the machine's extraction
      harmonic) is tracked with the exact drift-and-kick map.
    - ONE representative bunch (not the real two-bunch split/merge
      topology) is tracked.
    - The entire squeeze+merge sequence is compressed into a single fast
      RF-voltage ramp: Vrf_min -> Vrf_max over ~1000 turns. This stands in
      for "bucket gets narrower/higher-voltage," and is deliberately fast
      compared to the synchrotron period so that it is NON-adiabatic and
      therefore causes real emittance growth (a slow/adiabatic ramp would
      approximately conserve emittance and not test anything).

Physical picture (drift + kick), unchanged from the parametric-resonance
version this was adapted from:

    (1) DRIFT : exact relativistic revolution-time dependence on energy
        deviation dE (via momentum, velocity, and momentum-compaction
        orbit-length change) -- no linear eta*dp/p shortcut.
    (2) KICK  : exact nonlinear sinusoidal RF kick, stable phase at pi
        (above transition), single fixed harmonic.

What's measured
----------------
Full longitudinal emittance in eVs, using the "uniform-filled-ellipse"
convention eps_full = 4*pi*sigma_t[s]*sigma_dE[eV] (equivalently
pi*a_t*a_E for the ellipse's own semi-axes). This is a specific, stated
convention chosen for order-of-magnitude comparison -- it is explicitly
NOT validated against BNL's internal Bbat/Bbrat convention (that
reconciliation is out of scope here per project decision).

Validation target (from the note, within-process comparison):
    pre-squeeze proxy   eps ~ 1.09 eVs   (measurement 40 ms before flattop)
    post-merge          eps ~ 1.24 eVs   (Table I, operational/RHIC rows)
    => within-process growth ~ 14%

    (The note's separately-quoted "~25% growth, 1.24 vs 0.99 eVs" compares
    the squeeze/merge setup's final eps against the *standard AU4 (no
    squeeze) setup's* final eps -- a different-machine-configuration
    comparison, not a before/after of this process. That number is
    reported below for context only and is NOT the primary target.)

Pass/fail criterion used here: "ballpark," not a precision match --
absolute eps of order 1 eVs both before and after, and growth in the
tens-of-percent range, roughly consistent with ~14%.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import brentq

# =====================================================================
# COMMAND-LINE OVERRIDES (for fast testing)
# =====================================================================
parser = argparse.ArgumentParser(description="AGS squeeze/merge emittance-growth check")
parser.add_argument("--debug", action="store_true",
                     help="Use fewer particles/turns for a fast smoke test.")
parser.add_argument("--no-animation", action="store_true",
                     help="Skip mp4 rendering (diagnostics/CSV still produced).")
cli_args, _ = parser.parse_known_args()

# =====================================================================
# CONFIGURATION
# =====================================================================
RNG_SEED = 12345
np.random.seed(RNG_SEED)

DEBUG_MODE = bool(cli_args.debug)
MAKE_ANIMATION = not cli_args.no_animation

OUTPUT_CSV = "squeeze_merge_diagnostics.csv"
OUTPUT_MP4 = "squeeze_merge_animation.mp4"

# ---------------------------------------------------------------------
# Machine parameters (AGS)
# ---------------------------------------------------------------------
K0 = 24.0                  # reference kinetic energy [GeV]
mp = 0.938272               # proton rest mass [GeV]
c = 299792458.0             # speed of light [m/s]
L0 = 807.1                  # AGS circumference [m]

gamma_t = 8.45
alpha_p = 1.0 / gamma_t**2

h = 12                        # single representative RF harmonic

Vrf_max = 320e3 / 1e9        # 320 kV -> GeV  (post-merge / operational voltage)
Vrf_min = 160e3 / 1e9        # 160 kV -> GeV  (pre-squeeze voltage)

# ---------------------------------------------------------------------
# Squeeze/merge schedule: single fast monotonic voltage ramp
# ---------------------------------------------------------------------
squeeze_start_turn = 3000        # turn at which the fast ramp begins
squeeze_ramp_turns = 1000        # duration of the ramp [turns] -- deliberately
                                  # fast (non-adiabatic) compared to T_s (printed
                                  # below); this is the free knob if growth
                                  # comes out too small/large vs. the ~14% target
post_squeeze_settle_turns = 8000  # turns to let phase space filament/settle
                                    # after the ramp completes, before measuring
                                    # the "after" emittance

Vrf_start = Vrf_min
Vrf_end = Vrf_max

# ---------------------------------------------------------------------
# Target initial condition (matched, no artificial seed mismatch --
# the ramp itself is the perturbation under study)
# ---------------------------------------------------------------------
TARGET_INITIAL_EPS_EVS = 0.95   # pre-squeeze proxy from the note (40 ms
                                  # before flattop); see module docstring
                                  # for why this (not 0.47-0.81 or 0.99) is
                                  # used as the initial-condition target

# ---------------------------------------------------------------------
# Bunch / tracking configuration
# ---------------------------------------------------------------------
if DEBUG_MODE:
    # Scale the whole schedule down together so the ramp still falls inside
    # the shortened run (a fixed small n_turns with the full-scale squeeze_*
    # turns would silently skip the ramp entirely).
    N = 500
    squeeze_start_turn = 150
    squeeze_ramp_turns = 50
    post_squeeze_settle_turns = 300
    n_turns = squeeze_start_turn + squeeze_ramp_turns + post_squeeze_settle_turns
else:
    N = 10000
    n_turns = squeeze_start_turn + squeeze_ramp_turns + post_squeeze_settle_turns

# windows used to average "before" / "after" emittance (avoid edge transients)
_pre_margin = min(500, squeeze_start_turn // 4)
_post_margin = min(3000, post_squeeze_settle_turns // 2)
PRE_WINDOW = (_pre_margin, squeeze_start_turn - _pre_margin)
POST_WINDOW = (n_turns - _post_margin, n_turns - 1)

LOG_EVERY = 1
turns_per_frame = 10
MAX_FRAMES = 400 if not DEBUG_MODE else 50
fps = 30

# =====================================================================
# REFERENCE PARTICLE KINEMATICS
# =====================================================================
E0 = K0 + mp
p0 = np.sqrt(E0**2 - mp**2)
beta0 = p0 / E0
gamma0 = E0 / mp

if gamma0 <= gamma_t:
    raise ValueError(
        f"Reference particle (gamma0={gamma0:.4f}) is not above transition "
        f"(gamma_t={gamma_t}); this script assumes above-transition operation."
    )

T0 = L0 / (beta0 * c)
T0_ns = T0 * 1e9
T_rf = T0 / h
T_rf_ns = T_rf * 1e9

eta_slip = alpha_p - 1.0 / gamma0**2

print("=" * 70)
print("AGS REFERENCE PARTICLE / MACHINE PARAMETERS")
print("=" * 70)
print(f"  Total energy E0            = {E0:.6f} GeV")
print(f"  gamma0                     = {gamma0:.6f}   (gamma_t = {gamma_t})")
print(f"  Slip factor eta            = {eta_slip:.6e}  (expect > 0, above transition)")
print(f"  Revolution period T0       = {T0_ns:.6f} ns")
print(f"  RF period T_rf = T0/h      = {T_rf_ns:.6f} ns  (h = {h}, single representative harmonic)")
print()


# =====================================================================
# EXACT REVOLUTION TIME AND DRIFT MAP
# =====================================================================
def revolution_time(dE):
    E = E0 + dE
    p = np.sqrt(np.maximum(E**2 - mp**2, 1e-30))
    beta = p / E
    dp_over_p0 = (p - p0) / p0
    C = L0 * (1.0 + alpha_p * dp_over_p0)
    T = C / (beta * c)
    return T


def drift_map(t_ns, dE):
    T = revolution_time(dE)
    dt_ns = (T - T0) * 1e9
    return t_ns + dt_ns


def wrap_to_bucket(time_ns, T_rf_ns):
    return ((time_ns + T_rf_ns / 2) % T_rf_ns) - T_rf_ns / 2


# =====================================================================
# RF KICK (exact nonlinear sinusoidal form, above-transition convention)
# =====================================================================
def rf_kick(t_ns, dE, Vrf):
    """phi = 2*pi*h*t_ns/T0_ns + pi; the +pi shifts the stable phase to pi,
    correct above transition (verified below)."""
    phi = 2.0 * np.pi * h * t_ns / T0_ns + np.pi
    return dE + Vrf * np.sin(phi)


def one_turn_map(t_ns, dE, Vrf):
    t_new = drift_map(t_ns, dE)
    dE_new = rf_kick(t_new, dE, Vrf)
    return t_new, dE_new


# =====================================================================
# LINEARIZED COEFFICIENTS a, b AND FIXED-POINT STABILITY CHECK
# =====================================================================
def compute_a_coefficient():
    dE_step = 1.0e-6
    T_plus = revolution_time(dE_step)
    T_minus = revolution_time(-dE_step)
    dTdE = (T_plus - T_minus) / (2.0 * dE_step)
    return dTdE * 1e9


def compute_b_coefficient(Vrf_ref):
    return -Vrf_ref * 2.0 * np.pi * h / T0_ns


def check_fixed_point_stability(a, b):
    prod = a * b
    if prod >= 0:
        raise RuntimeError(
            f"Fixed point (t=0, dE=0) is NOT stable: a*b = {prod:.6e} >= 0."
        )
    print(f"Fixed-point stability check passed: a={a:.6e} ns/GeV, "
          f"b={b:.6e} GeV/ns, a*b={prod:.6e} < 0 (stable).")


a_coef = compute_a_coefficient()
# Use the mean of start/end voltage for the reference (a,b) used to size the
# initial matched bunch; the ramp itself uses the full-fidelity nonlinear map.
Vrf_ref_for_ab = Vrf_start   # bunch is matched pre-squeeze; the ramp to Vrf_end
                              # is what should knock it out of equilibrium
b_coef = compute_b_coefficient(Vrf_ref_for_ab)
check_fixed_point_stability(a_coef, b_coef)

omega_s_lin = np.sqrt(-a_coef * b_coef)
T_s_lin_turns = 2.0 * np.pi / omega_s_lin
f_synch_Hz = omega_s_lin / (2.0 * np.pi * T0)   # T0 is the revolution period in seconds
print('Synchronous frequency')
print(f_synch_Hz)

print()
print(f"  Synchrotron period at pre-squeeze Vrf ({Vrf_ref_for_ab*1e9/1e3:.0f} kV): "
      f"T_s ~ {T_s_lin_turns:.1f} turns")
print(f"  Squeeze ramp duration: {squeeze_ramp_turns} turns "
      f"({squeeze_ramp_turns / T_s_lin_turns:.2f} x T_s)")
if squeeze_ramp_turns / T_s_lin_turns > 3.0:
    warnings.warn(
        f"Ramp duration is {squeeze_ramp_turns / T_s_lin_turns:.1f} synchrotron "
        f"periods -- this is closer to the ADIABATIC regime and may under-"
        f"predict emittance growth relative to the note's fast squeeze/merge. "
        f"Consider shortening squeeze_ramp_turns if growth comes out too small."
    )
print()

# =====================================================================
# SQUEEZE/MERGE VOLTAGE SCHEDULE (single fast monotonic ramp)
# =====================================================================
def Vrf_of_turn(n):
    """RF voltage [GeV] at turn n: constant at Vrf_start, then a raised-
    cosine ramp up to Vrf_end over squeeze_ramp_turns starting at
    squeeze_start_turn, then held constant at Vrf_end. The raised-cosine
    shape avoids a discontinuous dV/dn at the ramp edges; the ramp is still
    fast on purpose (see T_s comparison above) -- only the *edges* are
    smoothed, not the overall speed."""
    if n < squeeze_start_turn:
        return Vrf_start
    tau = n - squeeze_start_turn
    if tau >= squeeze_ramp_turns:
        return Vrf_end
    ramp_frac = tau / squeeze_ramp_turns
    smooth = 0.5 * (1.0 - np.cos(np.pi * ramp_frac))   # 0 -> 1
    return Vrf_start + (Vrf_end - Vrf_start) * smooth


# =====================================================================
# INITIAL BUNCH: matched ellipse sized to hit TARGET_INITIAL_EPS_EVS
# =====================================================================
# For a uniformly-filled phase-space ellipse with semi-axes a_t [ns] and
# a_E [GeV], the ellipse's own area is the full emittance in these mixed
# units: eps_full = pi * a_t * a_E, and because ns*GeV = 1e-9 s * 1e9 eV,
# that product is numerically already in eVs -- no extra unit-conversion
# factor needed. The matched-ellipse relation a_E = a_t*sqrt(|b/a|) (linear
# theory, used only to size the INITIAL bunch; tracking itself is fully
# nonlinear) then gives:
#     eps_full = pi * a_t^2 * sqrt(|b/a|)  =>  a_t = sqrt(eps_full / (pi*sqrt(|b/a|)))
_slope = np.sqrt(abs(b_coef / a_coef))
a_t = np.sqrt(TARGET_INITIAL_EPS_EVS / (np.pi * _slope))     # ns
a_E = a_t * _slope                                             # GeV
eps_full_initial_check = np.pi * a_t * a_E

print(f"Initial matched bunch sized for eps_full = {TARGET_INITIAL_EPS_EVS:.3f} eVs:")
print(f"  a_t = {a_t:.3f} ns   a_E = {a_E*1e3:.4f} MeV")
print(f"  check: pi*a_t*a_E = {eps_full_initial_check:.4f} eVs")
print()

theta_init = 2.0 * np.pi * np.random.rand(N)
r_init = np.sqrt(np.random.rand(N))

time = a_t * r_init * np.cos(theta_init)     # ns
dE = a_E * r_init * np.sin(theta_init)        # GeV

time_init_for_color = time.copy()

# =====================================================================
# SEPARATRIX / HAMILTONIAN MACHINERY (for visualizing the reshaping bucket)
# =====================================================================
# Continuous-time Hamiltonian approximation to the drift-then-kick map, as
# before. NOTE: with Vrf now time-dependent, this is only meaningful as an
# "instantaneous frozen-Vrf bucket" snapshot at a given turn, not a strict
# invariant of the tracked map (the map itself is not exactly Hamiltonian
# once Vrf varies turn-to-turn) -- treat it as a visualization aid.
def _estimate_bucket_dE_scale():
    from scipy.integrate import quad
    Vrf_hi = Vrf_end * 1.2   # safety margin above the largest voltage used
    H_sep_hi = Vrf_hi * (T0_ns / (2.0 * np.pi * h))
    target_hi = 2.0 * H_sep_hi
    dE_trial = 0.2
    for _ in range(20):
        val, _ = quad(lambda e: (revolution_time(e) - T0) * 1e9, 0, dE_trial)
        if val >= target_hi:
            return dE_trial
        dE_trial *= 1.5
    return dE_trial


_dE_grid_max = _estimate_bucket_dE_scale()
_dE_grid = np.linspace(-_dE_grid_max, _dE_grid_max, 8001)
_integrand = (revolution_time(_dE_grid) - T0) * 1e9
_F_grid = np.concatenate([[0.0], np.cumsum(
    0.5 * (_integrand[1:] + _integrand[:-1]) * np.diff(_dE_grid))])
_zero_idx = np.argmin(np.abs(_dE_grid))
_F_grid = _F_grid - _F_grid[_zero_idx]


def F_of_dE(dE_val):
    return np.interp(dE_val, _dE_grid, _F_grid)


def G_of_t(t_ns, Vrf):
    return -Vrf * (T0_ns / (2.0 * np.pi * h)) * np.cos(2.0 * np.pi * h * t_ns / T0_ns)


def separatrix_H(Vrf):
    t_u = T0_ns / (2.0 * h)
    return G_of_t(t_u, Vrf)


def separatrix_dE(t_array, Vrf, dE_search_max=None):
    if dE_search_max is None:
        dE_search_max = 0.5 * _dE_grid_max

    H_sep = separatrix_H(Vrf)
    dE_pos = np.full_like(t_array, np.nan, dtype=float)
    dE_neg = np.full_like(t_array, np.nan, dtype=float)

    for i, t in enumerate(t_array):
        target = H_sep - G_of_t(t, Vrf)

        bracket_hi = dE_search_max
        found = False
        for _ in range(6):
            bracket_hi = min(bracket_hi, _dE_grid_max)
            if F_of_dE(bracket_hi) >= target:
                found = True
                break
            if bracket_hi >= _dE_grid_max:
                break
            bracket_hi *= 2.0
        if found and target >= 0.0:
            try:
                dE_pos[i] = brentq(lambda e: F_of_dE(e) - target, 0.0, bracket_hi)
            except ValueError:
                pass

        bracket_lo = -dE_search_max
        found = False
        for _ in range(6):
            bracket_lo = max(bracket_lo, -_dE_grid_max)
            if F_of_dE(bracket_lo) >= target:
                found = True
                break
            if bracket_lo <= -_dE_grid_max:
                break
            bracket_lo *= 2.0
        if found and target >= 0.0:
            try:
                dE_neg[i] = brentq(lambda e: F_of_dE(e) - target, bracket_lo, 0.0)
            except ValueError:
                pass

    return dE_pos, dE_neg


# =====================================================================
# MAIN TRACKING LOOP
# =====================================================================
diagnostics_rows = []
snapshot_times = []
snapshot_dEs = []
snapshot_turns = []
snapshot_Vrf = []

save_snapshot_every = max(1, n_turns // (turns_per_frame * MAX_FRAMES)) * turns_per_frame \
    if MAKE_ANIMATION else n_turns + 1

print("=" * 70)
print(f"RUNNING TRACKING: N={N} particles, n_turns={n_turns}, DEBUG_MODE={DEBUG_MODE}")
print(f"  squeeze ramp: turns [{squeeze_start_turn}, {squeeze_start_turn + squeeze_ramp_turns}] "
      f"  Vrf: {Vrf_start*1e9/1e3:.0f} -> {Vrf_end*1e9/1e3:.0f} kV")
print("=" * 70)

for n in range(n_turns):
    Vrf_n = Vrf_of_turn(n)

    T = revolution_time(dE)
    dt_ns = (T - T0) * 1e9
    time = time + dt_ns
    time = wrap_to_bucket(time, T_rf_ns)

    phi = 2.0 * np.pi * h * time / T0_ns + np.pi
    dE = dE + Vrf_n * np.sin(phi)

    if n % LOG_EVERY == 0:
        time_mean = np.mean(time)
        dE_mean = np.mean(dE)
        time_sigma_ns = np.sqrt(np.mean((time - time_mean)**2))
        dE_sigma_GeV = np.sqrt(np.mean((dE - dE_mean)**2))

        # RMS emittance-like quantity (ns*GeV), kept for continuity/debugging
        t2 = np.mean(time**2)
        dE2 = np.mean(dE**2)
        t_dE = np.mean(time * dE)
        eps_rms_ns_GeV = np.sqrt(max(t2 * dE2 - t_dE**2, 0.0))

        # Full emittance in eVs: eps_full = 4*pi*sigma_t[s]*sigma_dE[eV]
        # = 4*pi*sigma_t[ns]*sigma_dE[GeV] numerically (unit cancellation,
        # see docstring). This is the primary comparison quantity.
        eps_full_eVs = 4.0 * np.pi * time_sigma_ns * dE_sigma_GeV

        diagnostics_rows.append({
            "turn": n,
            "Vrf_kV": Vrf_n * 1e9 / 1e3,
            "time_sigma_ns": time_sigma_ns,
            "dE_sigma_MeV": dE_sigma_GeV * 1e3,
            "eps_rms_ns_GeV": eps_rms_ns_GeV,
            "eps_full_eVs": eps_full_eVs,
        })

    if MAKE_ANIMATION and (n % save_snapshot_every == 0):
        snapshot_times.append(time.copy())
        snapshot_dEs.append(dE.copy())
        snapshot_turns.append(n)
        snapshot_Vrf.append(Vrf_n)

print(f"Tracking complete. Logged {len(diagnostics_rows)} diagnostic rows, "
      f"{len(snapshot_turns)} animation snapshots.")
print()

df = pd.DataFrame(diagnostics_rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Diagnostics saved to: {OUTPUT_CSV}")

# =====================================================================
# BEFORE / AFTER EMITTANCE COMPARISON (the actual ask)
# =====================================================================
pre_mask = (df.turn >= PRE_WINDOW[0]) & (df.turn <= PRE_WINDOW[1])
post_mask = (df.turn >= POST_WINDOW[0]) & (df.turn <= POST_WINDOW[1])

eps_before = df.loc[pre_mask, "eps_full_eVs"].mean()
eps_after = df.loc[post_mask, "eps_full_eVs"].mean()
growth_factor = eps_after / eps_before
growth_pct = (growth_factor - 1.0) * 100.0

NOTE_PRE = 1.09     # pre-squeeze proxy, 40 ms before flattop
NOTE_POST = 1.24    # post-merge, operational (Table I)
NOTE_GROWTH_PCT = (NOTE_POST / NOTE_PRE - 1.0) * 100.0
NOTE_AU4_CONTROL = 0.99   # standard no-squeeze setup, for context only

print()
print("=" * 70)
print("EMITTANCE GROWTH CHECK vs. C-A/AP/685")
print("=" * 70)
print(f"  Simulated  eps_before = {eps_before:.3f} eVs  "
      f"(averaged turns {PRE_WINDOW[0]}-{PRE_WINDOW[1]})")
print(f"  Simulated  eps_after  = {eps_after:.3f} eVs  "
      f"(averaged turns {POST_WINDOW[0]}-{POST_WINDOW[1]})")
print(f"  Simulated  growth     = {growth_factor:.3f}x  ({growth_pct:+.1f}%)")
print()
print(f"  Note (within-process): {NOTE_PRE:.2f} -> {NOTE_POST:.2f} eVs  "
      f"({NOTE_GROWTH_PCT:+.1f}%)   <- primary comparison target")
print(f"  Note (cross-setup, context only): squeeze/merge {NOTE_POST:.2f} eVs "
      f"vs. standard AU4 {NOTE_AU4_CONTROL:.2f} eVs "
      f"({(NOTE_POST/NOTE_AU4_CONTROL - 1)*100:+.1f}%)")
print()

scale_ok = 0.3 < eps_before < 3.0 and 0.3 < eps_after < 3.0
growth_ok = 5.0 < growth_pct < 60.0   # loose "tens of percent" band
print(f"  Absolute scale O(1) eVs:      {'OK' if scale_ok else 'CHECK'}")
print(f"  Growth in tens-of-percent range: {'OK' if growth_ok else 'CHECK'}")
print("=" * 70)
print()

# =====================================================================
# ANIMATION
# =====================================================================
if MAKE_ANIMATION and len(snapshot_turns) > 0:
    print("Rendering animation...")

    fig, ax = plt.subplots(figsize=(8, 6))
    t_plot_lim = 20.0 * a_t
    dE_plot_lim_MeV = 20.0 * a_E * 1e3

    ax.set_xlim(-t_plot_lim, t_plot_lim)
    ax.set_ylim(-dE_plot_lim_MeV, dE_plot_lim_MeV)
    ax.set_xlabel("Time deviation [ns]")
    ax.set_ylabel("Energy deviation [MeV]")

    scat = ax.scatter([], [], c=[], cmap="twilight", s=3, vmin=-a_t, vmax=a_t)
    sep_pos_line, = ax.plot([], [], "r-", lw=1.5)
    sep_neg_line, = ax.plot([], [], "r-", lw=1.5)
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                         va="top", ha="left", fontsize=9, family="monospace",
                         bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    t_sep_array = np.linspace(-T_rf_ns / 2, T_rf_ns / 2, 200)

    def init_anim():
        scat.set_offsets(np.empty((0, 2)))
        sep_pos_line.set_data([], [])
        sep_neg_line.set_data([], [])
        info_text.set_text("")
        return scat, sep_pos_line, sep_neg_line, info_text

    def update_anim(frame_idx):
        t_snap = snapshot_times[frame_idx]
        dE_snap = snapshot_dEs[frame_idx]
        turn_snap = snapshot_turns[frame_idx]
        Vrf_snap = snapshot_Vrf[frame_idx]

        scat.set_offsets(np.column_stack([t_snap, dE_snap * 1e3]))
        scat.set_array(time_init_for_color)

        dE_pos, dE_neg = separatrix_dE(t_sep_array, Vrf_snap)
        sep_pos_line.set_data(t_sep_array, dE_pos * 1e3)
        sep_neg_line.set_data(t_sep_array, dE_neg * 1e3)

        if turn_snap < squeeze_start_turn:
            phase_label = "pre-squeeze"
        elif turn_snap < squeeze_start_turn + squeeze_ramp_turns:
            phase_label = "squeezing"
        else:
            phase_label = "post-merge (settling)"

        info_text.set_text(
            f"turn        = {turn_snap:d}\n"
            f"phase       = {phase_label}\n"
            f"Vrf         = {Vrf_snap*1e9/1e3:.1f} kV\n"
            f"T_s (est.)  = {T_s_lin_turns:.1f} turns"
        )
        return scat, sep_pos_line, sep_neg_line, info_text

    n_frames = len(snapshot_turns)
    anim = animation.FuncAnimation(
        fig, update_anim, frames=n_frames, init_func=init_anim,
        blit=False, interval=1000 / fps
    )

    try:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(OUTPUT_MP4, writer=writer)
        print(f"Animation saved to: {OUTPUT_MP4}")
    except Exception as exc:
        warnings.warn(f"Could not save animation (is ffmpeg installed?): {exc}")
    plt.close(fig)
else:
    print("Animation skipped (MAKE_ANIMATION=False or no snapshots recorded).")

# =====================================================================
# POST-RUN DIAGNOSTIC PLOTS
# =====================================================================
def save_plot(x, y, xlabel, ylabel, title, fname, hlines=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, lw=1.2)
    if hlines:
        for yv, lbl in hlines:
            ax.axhline(yv, color="gray", ls="--", lw=1, alpha=0.7)
            ax.text(x.iloc[-1] if hasattr(x, "iloc") else x[-1], yv, f" {lbl}",
                    va="center", ha="right", fontsize=8, color="gray")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  saved {fname}")


print()
print("Generating diagnostic plots...")
save_plot(df.turn, df.Vrf_kV, "Turn", "V_rf [kV]",
          "RF voltage vs. turn (squeeze/merge ramp)", "plot_1_Vrf_vs_turn.png")
save_plot(df.turn, df.time_sigma_ns, "Turn", "RMS bunch length [ns]",
          "RMS bunch length vs. turn", "plot_2_time_sigma_vs_turn.png")
save_plot(df.turn, df.dE_sigma_MeV, "Turn", "RMS energy spread [MeV]",
          "RMS energy spread vs. turn", "plot_3_dE_sigma_vs_turn.png")
save_plot(df.turn, df.eps_full_eVs, "Turn", "Full longitudinal emittance [eVs]",
          "Emittance vs. turn (dashed lines = note's pre/post values)",
          "plot_4_eps_full_eVs_vs_turn.png",
          hlines=[(NOTE_PRE, f"note pre={NOTE_PRE}"), (NOTE_POST, f"note post={NOTE_POST}")])

print()
print("=" * 70)
print("RUN COMPLETE")
print("=" * 70)