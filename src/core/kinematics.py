"""
Reference-particle kinematics and the exact drift map.

This is pulled directly from the "REFERENCE PARTICLE KINEMATICS" and
"EXACT REVOLUTION TIME AND DRIFT MAP" sections that used to be duplicated
at the top of every method script. It depends only on core.constants, so
adiabatic/non_adiabatic/resonant all import the same numbers.
"""

import numpy as np

from core.constants import K0, mp, c, L0, gamma_t, alpha_p, h

E0 = K0 + mp
p0 = np.sqrt(E0**2 - mp**2)
beta0 = p0 / E0
gamma0 = E0 / mp

if gamma0 <= gamma_t:
    raise ValueError(
        f"Reference particle (gamma0={gamma0:.4f}) is not above transition "
        f"(gamma_t={gamma_t}); this codebase assumes above-transition operation."
    )

T0 = L0 / (beta0 * c)          # revolution period [s]
T0_ns = T0 * 1e9
T_rf = T0 / h                    # RF period [s]
T_rf_ns = T_rf * 1e9

eta_slip = alpha_p - 1.0 / gamma0**2   # linear slip factor (>0 above transition)


def revolution_time(dE):
    """Exact revolution period [s] of a particle with energy deviation dE [GeV]."""
    E = E0 + dE
    p = np.sqrt(np.maximum(E**2 - mp**2, 1e-30))
    beta = p / E
    dp_over_p0 = (p - p0) / p0
    C = L0 * (1.0 + alpha_p * dp_over_p0)
    return C / (beta * c)


def drift_map(t_ns, dE):
    """One-turn drift: update arrival-time deviation t [ns] at fixed dE [GeV]."""
    T = revolution_time(dE)
    dt_ns = (T - T0) * 1e9
    return t_ns + dt_ns


def wrap_to_bucket(time_ns):
    """Wrap a time coordinate into (-T_rf_ns/2, +T_rf_ns/2]."""
    return ((time_ns + T_rf_ns / 2) % T_rf_ns) - T_rf_ns / 2


def print_summary():
    print("=" * 70)
    print("AGS REFERENCE PARTICLE / MACHINE PARAMETERS")
    print("=" * 70)
    print(f"  Total energy E0            = {E0:.6f} GeV")
    print(f"  gamma0                     = {gamma0:.6f}   (gamma_t = {gamma_t})")
    print(f"  Slip factor eta            = {eta_slip:.6e}")
    print(f"  Revolution period T0       = {T0_ns:.6f} ns")
    print(f"  RF period T_rf = T0/h      = {T_rf_ns:.6f} ns")
