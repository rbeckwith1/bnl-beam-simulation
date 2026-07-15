"""
RF kick (exact nonlinear form, above-transition convention: stable phase
is pi), the combined one-turn map, and the linearized a/b coefficients used
for the linear synchrotron-frequency estimate and matched-ellipse aspect
ratio. Any method-specific Vrf is passed in by the caller -- nothing here
hardcodes a particular voltage program.
"""

import numpy as np

from core.constants import h
from core.kinematics import T0_ns, revolution_time


def rf_kick(t_ns, dE, Vrf):
    """phi = 2*pi*h*t_ns/T0_ns + pi; the +pi is the above-transition stable
    phase (verified by check_fixed_point_stability)."""
    phi = 2.0 * np.pi * h * t_ns / T0_ns + np.pi
    return dE + Vrf * np.sin(phi)


def one_turn_map(t_ns, dE, Vrf):
    """DRIFT (using current dE) then KICK (using post-drift t)."""
    from core.kinematics import drift_map
    t_new = drift_map(t_ns, dE)
    dE_new = rf_kick(t_new, dE, Vrf)
    return t_new, dE_new


def compute_a_coefficient():
    """a in dt/dn = a*dE [ns/GeV], numerical derivative of revolution_time at dE=0."""
    dE_step = 1.0e-6
    T_plus = revolution_time(dE_step)
    T_minus = revolution_time(-dE_step)
    dTdE = (T_plus - T_minus) / (2.0 * dE_step)
    return dTdE * 1e9


def compute_b_coefficient(Vrf_ref):
    """b in ddE/dn = b*t [GeV/ns], linearizing the RF kick about (t=0, dE=0)."""
    return -Vrf_ref * 2.0 * np.pi * h / T0_ns


def check_fixed_point_stability(a, b):
    """Require a*b < 0 (omega_s^2 = -a*b > 0) for (t=0, dE=0) to be a stable center."""
    prod = a * b
    if prod >= 0:
        raise RuntimeError(
            f"Fixed point (t=0, dE=0) is NOT stable: a*b = {prod:.6e} >= 0. "
            f"Re-check the RF phase convention (phi offset)."
        )
    print(f"Fixed-point stability check passed: a={a:.6e} ns/GeV, "
          f"b={b:.6e} GeV/ns, a*b={prod:.6e} < 0 (stable).")
