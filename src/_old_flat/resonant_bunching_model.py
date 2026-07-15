"""
AGS resonant bunching simulation.

- Time: nanoseconds internally (t, T0_ns, T_rf_ns), consistent with the
  Vrf * sin(phase) kick already being in GeV per turn.
- All "expensive" one-time calculations (synchrotron frequency by two
  independent methods, matched-ellipse amplitudes, separatrix machinery
  setup) are done ONCE, before the main per-turn tracking loop.
- Above transition, the stable RF phase is pi (not 0); this is a matched
  fixed point and is verified numerically before being used (see
  `check_fixed_point_stability`).
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend by default; only switched if the animation
# actually needs a display, which it does not (we render straight to mp4).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import brentq

# =====================================================================
# COMMAND-LINE OVERRIDES (for fast testing)
# =====================================================================
parser = argparse.ArgumentParser(description="AGS resonant bunching simulation")
parser.add_argument("--debug", action="store_true",
                     help="Use fewer particles/turns for a fast smoke test.")
parser.add_argument("--no-animation", action="store_true",
                     help="Skip mp4 rendering (diagnostics/CSV still produced).")
cli_args, _ = parser.parse_known_args()

# =====================================================================
# CONFIGURATION  (all tunable knobs live here)
# =====================================================================
RNG_SEED = 12345
np.random.seed(RNG_SEED)

DEBUG_MODE = bool(cli_args.debug)          # fewer particles/turns, faster run
MAKE_ANIMATION = not cli_args.no_animation  # render mp4 animation or skip it

OUTPUT_CSV = "resonant_bunching_diagnostics.csv"
OUTPUT_MP4 = "resonant_bunching_animation.mp4"

# ---------------------------------------------------------------------
# Machine parameters (AGS), as given
# ---------------------------------------------------------------------
K0 = 24.0                  # reference kinetic energy [GeV]
mp = 0.938272               # proton rest mass [GeV]
c = 299792458.0             # speed of light [m/s]
L0 = 807.1                  # AGS circumference [m]

gamma_t = 8.45
alpha_p = 1.0 / gamma_t**2

h = 6                        # RF harmonic number

Vrf_max = 320e3 / 1e9        # 320 kV -> GeV
Vrf_min = 160e3 / 1e9        # 160 kV -> GeV

# ---------------------------------------------------------------------
# RF voltage modulation configuration
# ---------------------------------------------------------------------
Vrf_mean = 240e3 / 1e9       # mean RF voltage used for constant / modulated runs [GeV]
modulation_depth = 0.15      # fractional modulation depth (10-20% suggested)
resonance_ratio = 2.0        # omega_mod = resonance_ratio * omega_s * (1+detuning)
detuning = 0.0                # fractional detuning of the modulation frequency
modulation_start_turn = 2000     # turn at which modulation switches on
modulation_ramp_turns = 12000     # turns over which depth ramps 0 -> modulation_depth
                                # (0 = instantaneous switch-on; >0 = smooth ramp,
                                # which avoids injecting its own transient kick)
modulation_phase = 0.0        # modulation phase offset [rad] -- see note below

# NOTE on modulation_phase: the envelope orientation angle theta_Q evolves at
# 2*omega_s. Whether the voltage modulation AMPLIFIES or SUPPRESSES the
# envelope oscillation depends on the relative phase between cos(omega_mod*n
# + modulation_phase) and cos(2*theta_Q(n)). Shifting modulation_phase by
# pi/2 changes parametric driving into parametric damping (or vice versa);
# shifting by pi simply reverses which quadrature is driven. There is no
# a-priori reason modulation_phase = 0 is the "correct" phase for growth --
# this must be checked in the diagnostics (Q1, Q2, theta_Q vs. RF phase).

# ---------------------------------------------------------------------
# Initial bunch mismatch (optional envelope-oscillation seed)
# ---------------------------------------------------------------------
initial_time_mismatch = 1.05    # >1.0 makes the initial ellipse a monopole+quadrupole seed in t
initial_energy_mismatch = 1.0   # keep at 1.0 to isolate the effect of the time mismatch

# ---------------------------------------------------------------------
# Bunch / tracking configuration
# ---------------------------------------------------------------------
if DEBUG_MODE:
    N = 500
    n_turns = 500
else:
    N = 10000
    n_turns = 30000

# ---------------------------------------------------------------------
# Longitudinal emittance target (Run 24 reference note)
# ---------------------------------------------------------------------
# Convention: eps_l is the FULL matched-ellipse AREA, i.e. eps_l = pi*a_t*a_E,
# in units of ns*GeV. Note that ns*GeV == eV*s numerically (GeV = 1e9 eV,
# ns = 1e-9 s, so the two factors of 1e9 cancel exactly), so a Run 24 note
# quoting eps_l in eV*s can be entered here unchanged -- no unit conversion
# needed.
#
# TODO(Rosalyn): confirm with Dr. Brooks whether the BNL Bbat/Bbrat
# convention reports the FULL ellipse area, the area/pi ("normalized"), or
# a 4*sigma area -- each implies a different prefactor relative to what's
# used below (this derivation assumes eps_l = full ellipse area).
eps_l_ns_GeV = 0.95*(2/3)  # ns*GeV == eV*s -- PLACEHOLDER: replace with Run 24 value

# a_t (and a_E) are NOT picked directly -- they are DERIVED further below,
# once a_coef/b_coef (the linear ellipse aspect ratio) are known, from the
# requirement that the matched ellipse encloses exactly eps_l_ns_GeV.

LOG_EVERY = 1                  # log diagnostics every this many turns
turns_per_frame = 10            # animation: turns advanced per rendered frame
MAX_FRAMES = 400 if not DEBUG_MODE else 50   # cap memory use for animation buffer
fps = 30

# =====================================================================
# REFERENCE PARTICLE KINEMATICS
# =====================================================================
E0 = K0 + mp                          # total energy [GeV]
p0 = np.sqrt(E0**2 - mp**2)           # momentum [GeV/c]
beta0 = p0 / E0
gamma0 = E0 / mp

if gamma0 <= gamma_t:
    raise ValueError(
        f"Reference particle (gamma0={gamma0:.4f}) is not above transition "
        f"(gamma_t={gamma_t}); this script assumes above-transition operation."
    )

T0 = L0 / (beta0 * c)                 # reference revolution period [s]
T0_ns = T0 * 1e9
T_rf = T0 / h                          # RF period [s]
T_rf_ns = T_rf * 1e9

eta_slip = alpha_p - 1.0 / gamma0**2   # linear slip factor (should be > 0, above transition)

print("=" * 70)
print("AGS REFERENCE PARTICLE / MACHINE PARAMETERS")
print("=" * 70)
print(f"  Total energy E0            = {E0:.6f} GeV")
print(f"  Momentum p0                = {p0:.6f} GeV/c")
print(f"  beta0                      = {beta0:.10f}")
print(f"  gamma0                     = {gamma0:.6f}   (gamma_t = {gamma_t})")
print(f"  Momentum compaction alpha_p= {alpha_p:.6e}")
print(f"  Slip factor eta            = {eta_slip:.6e}  (expect > 0, above transition)")
print(f"  Revolution period T0       = {T0_ns:.6f} ns")
print(f"  RF period T_rf = T0/h      = {T_rf_ns:.6f} ns")
print()


# =====================================================================
# EXACT REVOLUTION TIME AND DRIFT MAP
# =====================================================================
def revolution_time(dE):
    """
    Exact revolution period [s] of a particle with energy deviation dE [GeV]
    relative to the reference particle.

    Includes:
      - exact relativistic momentum/velocity from total energy
      - orbit-length change from momentum compaction: C = L0*(1+alpha_p*dp/p0)
    No linear "eta*dp/p" shortcut is used; T is computed from first
    principles for every dE.
    """
    E = E0 + dE
    p = np.sqrt(np.maximum(E**2 - mp**2, 1e-30))
    beta = p / E
    dp_over_p0 = (p - p0) / p0
    C = L0 * (1.0 + alpha_p * dp_over_p0)   # orbit circumference [m]
    T = C / (beta * c)                        # revolution period [s]
    return T


def drift_map(t_ns, dE):
    """One-turn drift: update arrival-time deviation t [ns] using the exact
    revolution time as a function of the (unchanged-by-drift) energy
    deviation dE [GeV]."""
    T = revolution_time(dE)
    dt_ns = (T - T0) * 1e9
    return t_ns + dt_ns


def wrap_to_bucket(time_ns, T_rf_ns):
    """Wrap a time coordinate into (-T_rf_ns/2, +T_rf_ns/2]."""
    return ((time_ns + T_rf_ns / 2) % T_rf_ns) - T_rf_ns / 2


# =====================================================================
# RF KICK (exact nonlinear sinusoidal form, above-transition convention)
# =====================================================================
def rf_kick(t_ns, dE, Vrf):
    """
    One-turn RF kick using the exact nonlinear sinusoidal RF voltage.

    phi = 2*pi*h*t_ns/T0_ns + pi

    The "+pi" shifts the stable synchronous phase to pi, which is the
    correct convention above transition (see check_fixed_point_stability
    below for the numerical verification of why this sign/phase choice is
    stable and is therefore kept unchanged from the suggested form).
    """
    phi = 2.0 * np.pi * h * t_ns / T0_ns + np.pi
    return dE + Vrf * np.sin(phi)


def one_turn_map(t_ns, dE, Vrf):
    """Exact one-turn map: DRIFT (using current dE) then KICK (using the
    post-drift t). This ordering is used consistently everywhere in this
    script (tracking loop, Jacobian, and is noted as an approximation when
    building the continuous separatrix Hamiltonian below)."""
    t_new = drift_map(t_ns, dE)
    dE_new = rf_kick(t_new, dE, Vrf)
    return t_new, dE_new


# =====================================================================
# LINEARIZED COEFFICIENTS a, b  AND FIXED-POINT STABILITY CHECK
# =====================================================================
def compute_a_coefficient():
    """a in dt/dn = a*dE [ns/GeV], from numerical derivative of the exact
    revolution time w.r.t. energy deviation, evaluated at dE=0."""
    dE_step = 1.0e-6  # GeV
    T_plus = revolution_time(dE_step)
    T_minus = revolution_time(-dE_step)
    dTdE = (T_plus - T_minus) / (2.0 * dE_step)   # s / GeV
    return dTdE * 1e9                               # ns / GeV


def compute_b_coefficient(Vrf_ref):
    """b in ddE/dn = b*t [GeV/ns], from linearizing the RF kick about the
    (t=0, dE=0) fixed point:
        sin(2*pi*h*t/T0_ns + pi) = -sin(2*pi*h*t/T0_ns) ~= -(2*pi*h/T0_ns)*t
    => ddE/dn ~= -Vrf_ref*(2*pi*h/T0_ns)*t
    """
    return -Vrf_ref * 2.0 * np.pi * h / T0_ns


def check_fixed_point_stability(a, b):
    """Verify a*b < 0, i.e. omega_s^2 = -a*b > 0, which is required for the
    (t=0, dE=0) fixed point to be a stable (elliptic) center rather than a
    saddle. Raises with a clear message if the convention is unstable."""
    prod = a * b
    if prod >= 0:
        raise RuntimeError(
            f"Fixed point (t=0, dE=0) is NOT stable with the chosen phase "
            f"convention: a*b = {prod:.6e} >= 0 (need a*b < 0). "
            f"Re-check the RF phase convention (phi offset)."
        )
    print(f"Fixed-point stability check passed: a={a:.6e} ns/GeV, "
          f"b={b:.6e} GeV/ns, a*b={prod:.6e} < 0 (stable).")


a_coef = compute_a_coefficient()
b_coef = compute_b_coefficient(Vrf_mean)
check_fixed_point_stability(a_coef, b_coef)

# =====================================================================
# SYNCHROTRON FREQUENCY -- METHOD 1: linear (a,b) estimate
# =====================================================================
omega_s_lin = np.sqrt(-a_coef * b_coef)     # rad / turn
T_s_lin_turns = 2.0 * np.pi / omega_s_lin    # turns

# =====================================================================
# SYNCHROTRON FREQUENCY -- METHOD 2: Jacobian of the exact one-turn map
# =====================================================================
def compute_exact_jacobian(Vrf_ref):
    """Numerically linearize the EXACT one_turn_map about (t=0, dE=0) via
    central finite differences, and return the 2x2 Jacobian."""
    eps_t = 1.0e-3   # ns
    eps_E = 1.0e-8   # GeV

    def m(t_ns, dE):
        return one_turn_map(t_ns, dE, Vrf_ref)

    t_p, dE_p = m(eps_t, 0.0)
    t_m, dE_m = m(-eps_t, 0.0)
    dtdt = (t_p - t_m) / (2 * eps_t)
    ddEdt = (dE_p - dE_m) / (2 * eps_t)

    t_pE, dE_pE = m(0.0, eps_E)
    t_mE, dE_mE = m(0.0, -eps_E)
    dtdE = (t_pE - t_mE) / (2 * eps_E)
    ddEdE = (dE_pE - dE_mE) / (2 * eps_E)

    J = np.array([[dtdt, dtdE],
                  [ddEdt, ddEdE]])
    return J


J_exact = compute_exact_jacobian(Vrf_mean)
eigvals, eigvecs = np.linalg.eig(J_exact)

# For stable (elliptic) motion, eigenvalues are a complex-conjugate pair on
# the unit circle: lambda = exp(+-i*mu_s).
if np.any(np.abs(np.abs(eigvals) - 1.0) > 1e-3):
    warnings.warn(
        f"Exact one-turn-map Jacobian eigenvalues are not on the unit "
        f"circle (|lambda| = {np.abs(eigvals)}); the linear approximation "
        f"of the exact map may be inaccurate, or the fixed point is not "
        f"stable at this working point."
    )

mu_s = np.abs(np.angle(eigvals[0]))
if mu_s == 0:
    raise RuntimeError("Jacobian eigenvalues are purely real; fixed point "
                        "is not oscillatory (not elliptic) -- cannot define "
                        "a synchrotron period this way.")
T_s_jac_turns = 2.0 * np.pi / mu_s
omega_s_jac = mu_s

print()
print("=" * 70)
print("SYNCHROTRON FREQUENCY / PERIOD -- METHOD COMPARISON (before tracking)")
print("=" * 70)
print(f"  Method 1 (linear a,b):        omega_s = {omega_s_lin:.6e} rad/turn, "
      f"T_s = {T_s_lin_turns:.3f} turns")
print(f"  Method 2 (exact map Jacobian): omega_s = {omega_s_jac:.6e} rad/turn, "
      f"T_s = {T_s_jac_turns:.3f} turns")
print(f"  Relative difference in T_s: "
      f"{abs(T_s_jac_turns - T_s_lin_turns) / T_s_jac_turns * 100:.3f} %")
print()

# Use the exact (Jacobian) estimate as the working omega_s for the
# resonance condition, since it captures the exact drift map (Method 1 is
# used only as an independent cross-check and for the matched-ellipse
# amplitude relation below).
omega_s = omega_s_jac
T_s_turns = T_s_jac_turns

# =====================================================================
# RESONANT VOLTAGE MODULATION SETUP
# =====================================================================
omega_mod = resonance_ratio * omega_s * (1.0 + detuning)     # rad/turn
modulation_period_turns = 2.0 * np.pi / omega_mod              # turns

print(f"Modulation target: omega_mod = {resonance_ratio}*omega_s*(1+{detuning}) "
      f"= {omega_mod:.6e} rad/turn")
print(f"Modulation period: {modulation_period_turns:.3f} turns "
      f"(compare to T_s/2 = {T_s_turns/2:.3f} turns)")
print()


def modulation_depth_at_turn(n):
    """
    Effective modulation depth at turn n.

    Ramps smoothly from 0 to modulation_depth over modulation_ramp_turns
    turns after modulation_start_turn, using a half-cosine (raised-cosine)
    profile so dV/dn has no discontinuity at the start or end of the ramp.
    Setting modulation_ramp_turns = 0 recovers the original instantaneous
    switch-on.

    A hard instantaneous voltage step is itself a broadband excitation that
    can seed its own transient envelope oscillation -- ramping in the depth
    lets you isolate growth that is genuinely driven by the steady-state
    2*omega_s parametric resonance, rather than by the switch-on transient.
    """
    if n < modulation_start_turn:
        return 0.0
    tau = n - modulation_start_turn
    if modulation_ramp_turns <= 0:
        return modulation_depth
    ramp_frac = min(tau / modulation_ramp_turns, 1.0)
    smooth = 0.5 * (1.0 - np.cos(np.pi * ramp_frac))   # 0 -> 1, zero slope at both ends
    return modulation_depth * smooth


def Vrf_of_turn(n):
    """RF voltage [GeV] at turn n, per the configured modulation."""
    if n < modulation_start_turn:
        V = Vrf_mean
    else:
        tau = n - modulation_start_turn
        depth_n = modulation_depth_at_turn(n)
        V = Vrf_mean * (1.0 + depth_n *
                         np.cos(omega_mod * tau + modulation_phase))
    return float(np.clip(V, 1.0e-9, None))   # keep strictly positive/physical


# =====================================================================
# MATCHED-ELLIPSE AMPLITUDE RELATION AND INITIAL BUNCH
# =====================================================================
# The linear ellipse ASPECT RATIO a_E/a_t = sqrt(|b/a|) is fixed by the
# machine (independent of ellipse size), using the LINEAR (a,b)
# coefficients -- this is itself a linear-theory result used only to
# *initialize* the bunch shape; subsequent tracking is fully nonlinear.
#
# The ellipse SIZE is then fixed by requiring its area to equal the target
# emittance: eps_l = pi * a_t * a_E = pi * a_t^2 * (a_E/a_t)
#   => a_t = sqrt( eps_l / (pi * (a_E/a_t)) )
aspect_E_over_t = np.sqrt(abs(b_coef / a_coef))     # GeV/ns, fixed by machine
a_t = np.sqrt(eps_l_ns_GeV / (np.pi * aspect_E_over_t))   # ns
a_E = a_t * aspect_E_over_t                                 # GeV

_area_check = np.pi * a_t * a_E
print(f"Matched-ellipse initial amplitudes: a_t = {a_t:.3f} ns, "
      f"a_E = {a_E*1e3:.4f} MeV")
print(f"  -> enclosed area = pi*a_t*a_E = {_area_check:.6e} ns*GeV "
      f"(target eps_l = {eps_l_ns_GeV:.6e} ns*GeV)")
print(f"Initial mismatch factors: time x{initial_time_mismatch}, "
      f"energy x{initial_energy_mismatch}")
print()

theta_init = 2.0 * np.pi * np.random.rand(N)
r_init = np.sqrt(np.random.rand(N))

time = initial_time_mismatch * a_t * r_init * np.cos(theta_init)     # ns
dE = initial_energy_mismatch * a_E * r_init * np.sin(theta_init)      # GeV

time_init_for_color = time.copy()   # stored for particle coloring in the animation

# =====================================================================
# SEPARATRIX / HAMILTONIAN MACHINERY
# =====================================================================
# Continuous-time Hamiltonian approximation to the drift-then-kick map:
#
#   dt/dn  =  dF/ddE  =  [T(dE) - T0] * 1e9                (matches drift_map)
#   ddE/dn = -dG/dt
#
# Choosing G(t, Vrf) = -Vrf*(T0_ns/(2*pi*h))*cos(2*pi*h*t/T0_ns) reproduces
#   -dG/dt = Vrf*sin(2*pi*h*t/T0_ns) = -Vrf*sin(phi)   [phi = ...+ pi]
#          = Vrf*sin(phi)   (using sin(x+pi) = -sin(x) twice)
# i.e. exactly the RF kick used above.
#
# LIMITATION: this is a continuous (Hamiltonian-flow) approximation to a
# discrete drift-THEN-kick map. It is exact only in the limit of small
# per-turn phase advance; for finite per-turn kicks the true discrete map is
# not exactly symplectic-Hamiltonian with this H, so the separatrix drawn
# here is an approximation (standard in accelerator physics) rather than an
# exact invariant of the tracked map.

# --- F(dE): build once on a fine grid, then use fast interpolation ------
# The grid must safely exceed the true RF-bucket half-height in energy (NOT
# just the initial bunch amplitude a_E), since np.interp silently clamps
# outside its range -- an under-sized grid would silently truncate the
# separatrix rather than raising an error. The bucket half-height at the
# largest voltage used anywhere in this script (Vrf_max, further inflated
# by (1+modulation_depth)) is estimated via a coarse search and padded by a
# comfortable safety factor.
def _estimate_bucket_dE_scale():
    from scipy.integrate import quad
    Vrf_hi = Vrf_max * (1.0 + modulation_depth) * 1.2   # safety margin
    H_sep_hi = Vrf_hi * (T0_ns / (2.0 * np.pi * h))
    target_hi = 2.0 * H_sep_hi   # max possible target, at t=0
    dE_trial = 0.2
    for _ in range(20):
        val, _ = quad(lambda e: (revolution_time(e) - T0) * 1e9, 0, dE_trial)
        if val >= target_hi:
            return dE_trial
        dE_trial *= 1.5
    return dE_trial


_dE_grid_max = _estimate_bucket_dE_scale()
_dE_grid = np.linspace(-_dE_grid_max, _dE_grid_max, 8001)
_integrand = (revolution_time(_dE_grid) - T0) * 1e9        # dF/ddE, ns
_F_grid = np.concatenate([[0.0], np.cumsum(
    0.5 * (_integrand[1:] + _integrand[:-1]) * np.diff(_dE_grid))])
# _F_grid[i] = F(_dE_grid[i]) - F(_dE_grid[0]); shift so F(dE=0) = 0
_zero_idx = np.argmin(np.abs(_dE_grid))
_F_grid = _F_grid - _F_grid[_zero_idx]


def F_of_dE(dE_val):
    """F(dE) such that dF/ddE = [T(dE)-T0]*1e9, via fast lookup on a
    precomputed fine grid (F(0) = 0, F >= 0 everywhere for this above-
    transition machine)."""
    return np.interp(dE_val, _dE_grid, _F_grid)


def G_of_t(t_ns, Vrf):
    """RF potential term consistent with the kick convention (see derivation
    above)."""
    return -Vrf * (T0_ns / (2.0 * np.pi * h)) * np.cos(2.0 * np.pi * h * t_ns / T0_ns)


def separatrix_H(Vrf):
    """Hamiltonian value at the unstable fixed point t_u = T0_ns/(2h), dE=0
    (edge of the RF bucket, where dG/dt=0 and F(0)=0)."""
    t_u = T0_ns / (2.0 * h)
    return G_of_t(t_u, Vrf)


def separatrix_dE(t_array, Vrf, dE_search_max=None):
    """
    Solve H(t, dE) = F(dE) + G(t, Vrf) = H_sep for dE, given t, returning the
    positive- and negative-energy branches. Uses scipy.optimize.brentq with
    an automatically expanding bracket; explicitly reports (does not
    silently hide) any point where no bracket / root can be found.
    """
    if dE_search_max is None:
        dE_search_max = 0.5 * _dE_grid_max   # stay within the interpolation grid

    H_sep = separatrix_H(Vrf)
    dE_pos = np.full_like(t_array, np.nan, dtype=float)
    dE_neg = np.full_like(t_array, np.nan, dtype=float)

    for i, t in enumerate(t_array):
        target = H_sep - G_of_t(t, Vrf)   # solve F(dE) = target

        # --- positive branch: F is monotonic increasing on [0, dE_max] ---
        bracket_hi = dE_search_max
        found = False
        for _ in range(6):   # expand bracket if needed (never past the valid grid)
            bracket_hi = min(bracket_hi, _dE_grid_max)
            if F_of_dE(bracket_hi) >= target:
                found = True
                break
            if bracket_hi >= _dE_grid_max:
                break   # already at grid edge, cannot expand further
            bracket_hi *= 2.0
        if found and target >= 0.0:
            try:
                dE_pos[i] = brentq(lambda e: F_of_dE(e) - target, 0.0, bracket_hi)
            except ValueError as exc:
                warnings.warn(f"Brent root-finding failed for separatrix "
                               f"positive branch at t={t:.3f} ns: {exc}")
        elif target > F_of_dE(_dE_grid_max):
            warnings.warn(f"Separatrix positive branch at t={t:.3f} ns "
                           f"requires dE beyond the precomputed F(dE) grid "
                           f"(+-{_dE_grid_max:.4f} GeV); increase the grid "
                           f"range rather than trust this point.")
        # (target < 0 at this t simply means the bucket is closed there --
        # not an error.)

        # --- negative branch: search dE in [-bracket_hi, 0] -------------
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
            except ValueError as exc:
                warnings.warn(f"Brent root-finding failed for separatrix "
                               f"negative branch at t={t:.3f} ns: {exc}")
        elif target > F_of_dE(-_dE_grid_max):
            warnings.warn(f"Separatrix negative branch at t={t:.3f} ns "
                           f"requires dE beyond the precomputed F(dE) grid "
                           f"(+-{_dE_grid_max:.4f} GeV); increase the grid "
                           f"range rather than trust this point.")

    return dE_pos, dE_neg


# =====================================================================
# MAIN TRACKING LOOP
# =====================================================================
diagnostics_rows = []
snapshot_times = []
snapshot_dEs = []
snapshot_turns = []
snapshot_Vrf = []

n_frame_stride = turns_per_frame
save_snapshot_every = max(1, n_turns // (turns_per_frame * MAX_FRAMES)) * turns_per_frame \
    if MAKE_ANIMATION else n_turns + 1   # never trigger if animation disabled

print("=" * 70)
print(f"RUNNING TRACKING: N={N} particles, n_turns={n_turns}, "
      f"DEBUG_MODE={DEBUG_MODE}")
print("=" * 70)

for n in range(n_turns):
    Vrf_n = Vrf_of_turn(n)

    # --- drift ---
    T = revolution_time(dE)
    dt_ns = (T - T0) * 1e9
    time = time + dt_ns
    time = wrap_to_bucket(time, T_rf_ns)

    # --- kick (exact nonlinear sinusoidal RF) ---
    phi = 2.0 * np.pi * h * time / T0_ns + np.pi
    dE = dE + Vrf_n * np.sin(phi)

    # --- diagnostics ---
    if n % LOG_EVERY == 0:
        t2 = np.mean(time**2)
        dE2 = np.mean(dE**2)
        t_dE = np.mean(time * dE)

        time_sigma = np.sqrt(np.mean((time - np.mean(time))**2))
        dE_sigma_GeV = np.sqrt(np.mean((dE - np.mean(dE))**2))

        eps_rms = np.sqrt(max(t2 * dE2 - t_dE**2, 0.0))

        Q1 = np.mean((time / a_t)**2 - (dE / a_E)**2)
        Q2 = 2.0 * np.mean((time / a_t) * (dE / a_E))
        Q_amp = np.sqrt(Q1**2 + Q2**2)
        theta_Q = 0.5 * np.arctan2(Q2, Q1)

        diagnostics_rows.append({
            "turn": n,
            "Vrf_kV": Vrf_n * 1e9 / 1e3,
            "time_mean_ns": np.mean(time),
            "time_sigma_ns": time_sigma,
            "time_min_ns": np.min(time),
            "time_max_ns": np.max(time),
            "dE_mean_MeV": np.mean(dE) * 1e3,
            "dE_sigma_MeV": dE_sigma_GeV * 1e3,
            "dE_min_MeV": np.min(dE) * 1e3,
            "dE_max_MeV": np.max(dE) * 1e3,
            "t2": t2,
            "dE2": dE2,
            "t_dE": t_dE,
            "eps_rms": eps_rms,
            "Q1": Q1,
            "Q2": Q2,
            "Q_amp": Q_amp,
            "theta_Q": theta_Q,
        })

    # --- animation snapshot ---
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

    scat = ax.scatter([], [], c=[], cmap="twilight", s=3,
                       vmin=-a_t, vmax=a_t)
    sep_pos_line, = ax.plot([], [], "r-", lw=1.5)
    sep_neg_line, = ax.plot([], [], "r-", lw=1.5)
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                         va="top", ha="left", fontsize=9,
                         family="monospace",
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

        mod_phase_now = (omega_mod * max(turn_snap - modulation_start_turn, 0)
                          + modulation_phase) % (2 * np.pi)
        depth_now = modulation_depth_at_turn(turn_snap)

        info_text.set_text(
            f"turn        = {turn_snap:d}\n"
            f"Vrf         = {Vrf_snap*1e9/1e3:.1f} kV\n"
            f"mod. depth  = {depth_now:.3f}\n"
            f"mod. phase  = {mod_phase_now:.2f} rad\n"
            f"T_s (est.)  = {T_s_turns:.1f} turns\n"
            f"T_mod       = {modulation_period_turns:.1f} turns"
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
# POST-RUN DIAGNOSTIC PLOTS (separate figures, not subplots)
# =====================================================================
def save_plot(x, y, xlabel, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, lw=1.2)
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
          "RF voltage vs. turn", "plot_1_Vrf_vs_turn.png")
save_plot(df.turn, df.time_sigma_ns, "Turn", "RMS bunch length [ns]",
          "RMS bunch length vs. turn", "plot_2_time_sigma_vs_turn.png")
save_plot(df.turn, df.dE_sigma_MeV, "Turn", "RMS energy spread [MeV]",
          "RMS energy spread vs. turn", "plot_3_dE_sigma_vs_turn.png")
save_plot(df.turn, df.eps_rms, "Turn", "RMS emittance-like quantity [ns*GeV]",
          "RMS longitudinal emittance-like quantity vs. turn",
          "plot_4_eps_rms_vs_turn.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(df.turn, df.Q1, label="Q1")
ax.plot(df.turn, df.Q2, label="Q2")
ax.set_xlabel("Turn")
ax.set_ylabel("Quadrupole moments (normalized)")
ax.set_title("Q1, Q2 vs. turn")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plot_5_Q1_Q2_vs_turn.png", dpi=140)
plt.close(fig)
print("  saved plot_5_Q1_Q2_vs_turn.png")

save_plot(df.turn, df.Q_amp, "Turn", "Q_amp",
          "Quadrupole amplitude vs. turn", "plot_6_Qamp_vs_turn.png")
save_plot(df.turn, df.theta_Q, "Turn", "theta_Q [rad]",
          "Quadrupole orientation vs. turn", "plot_7_thetaQ_vs_turn.png")

mod_phase_series = np.where(
    df.turn >= modulation_start_turn,
    (omega_mod * (df.turn - modulation_start_turn) + modulation_phase) % (2 * np.pi),
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
fig.savefig("plot_8_phase_comparison.png", dpi=140)
plt.close(fig)
print("  saved plot_8_phase_comparison.png")

print()
print("=" * 70)
print("RUN COMPLETE")
print("=" * 70)
print(f"Final RMS bunch length: {df.time_sigma_ns.iloc[-1]:.3f} ns "
      f"(started at {df.time_sigma_ns.iloc[0]:.3f} ns)")
print(f"Final Q_amp: {df.Q_amp.iloc[-1]:.4f} (started at {df.Q_amp.iloc[0]:.4f})")